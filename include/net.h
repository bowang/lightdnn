#ifndef NET_H_
#define NET_H_

#include <vector>
#include "layer.h"

namespace lightdnn
{

// NOTE: currently, class Net only supports auto-encoder
template <class DataType>
class Net
{
  public:
    typedef Vector<DataType> VectorType;
    typedef Matrix<DataType> MatrixType;
    typedef Layer<DataType> LayerType;
    typedef typename LayerType::LayerParameter LayerParameter;

    struct NetParameter
    {
      unsigned batchSize;
      DataType alpha;
      DataType beta;
      DataType rho;
      DataType lambda;
    };

    Net (const NetParameter &param);

    virtual ~Net ();

    /*
     * @brief Add a new layer into the network
     */
    void addLayer (const LayerParameter &param);

    /*
     * @brief Feed-forward a sample
     */
    const VectorType& forward (const VectorType& x);

    /*
     * @brief Back-propagate a sample
     */
    void backward (const VectorType& x, const VectorType& y);

    /*
     * @brief Feed-forward a list of samples; cache the intermediate results
     */
    DataType batchForward (const MatrixType& X);

    /*
     * @brief Back-propagate with the cached intermediate results and a list of reference outputs
     */
    void batchBackward (const MatrixType& X, const MatrixType& Y);

    /*
     * @brief Calculate average activation
     */
    void averageActivation (unsigned m);

    /*
     * @brief Calculate the delta of weight and bias
     */
    void deltaParameters (unsigned m);

    /*
     * @brief Update the weights and bias of the network
     */
    void updateParameters ();

    /*
     * @brief Update the weights and bias of a layer
     */
    void update (uint64_t layerId, const MatrixType& weight, const VectorType& bias);

    /*
     * @brief Return the weight cost of the current model
     */
    DataType weightCost () const;

    /*
     * @brief Return the activation cost of the current model
     */
    DataType rhoCost () const;

    /*
     * @brief Get the number of layers
     */
    uint64_t numLayers () const { return layers_.size(); }

    /*
     * @brief Get the specified layer
     */
    LayerType& layer (int layerIdx) { return layers_[layerIdx]; }

    /*
     * @brief Get the delta weight (dW)
     */
    MatrixType& dW (int layerIdx) { return dW_[layerIdx]; }

    /*
     * @brief Get the delta bias (db)
     */
    VectorType& db (int layerIdx) { return db_[layerIdx]; }

  protected:
    const VectorType& forward (const VectorType& x,
        std::vector<VectorType>& z, std::vector<VectorType>& a);

    void backward (const VectorType& x, const VectorType& y,
        std::vector<VectorType>& z, std::vector<VectorType>& a);

    uint64_t batchSize_;
    DataType alpha_;
    DataType beta_;
    DataType rho_;
    DataType lambda_;

    std::vector<LayerType> layers_;           // layer[layerIdx]
    std::vector<std::vector<VectorType>> a_;  // activation[batchIdx][layerIdx]
    std::vector<std::vector<VectorType>> z_;  // linear_response[batchIdx][layerIdx]
    std::vector<MatrixType> dW_;              // delta_weight[layerIdx]
    std::vector<VectorType> db_;              // delta_bias[layerIdx]
    std::vector<VectorType> delta_;           // delta[layerIdx]
    std::vector<VectorType> a_sum_;           // sum_of_activations[layerIdx]
    std::vector<VectorType> a_avg_;           // average_of_activations[layerIdx]
};

template<class DataType>
inline Net<DataType>::Net (const NetParameter &param) :
    batchSize_ (param.batchSize), alpha_ (param.alpha), beta_ (param.beta),
    rho_ (param.rho), lambda_ (param.lambda)
{
  a_.resize (batchSize_);
  z_.resize (batchSize_);
}

template<class DataType>
inline Net<DataType>::~Net ()
{
}

template<class DataType>
inline void Net<DataType>::addLayer (const LayerParameter& param)
{
  layers_.emplace_back (param);

  for (uint64_t b = 0; b < batchSize_; b++) {
    a_[b].emplace_back (param.height);
    z_[b].emplace_back (param.height);
  }

  dW_.emplace_back (param.height, param.width, true);
  db_.emplace_back (param.height, true);
  delta_.emplace_back (param.height);
  a_sum_.emplace_back (param.height, true);
  a_avg_.emplace_back (param.height);
}

template<class DataType>
inline const Vector<DataType>& Net<DataType>::forward (const VectorType& x)
{
  return forward (x, z_[0], a_[0]);
}

template<class DataType>
inline const Vector<DataType>& Net<DataType>::forward (const VectorType& x,
    std::vector<VectorType>& z, std::vector<VectorType>& a)
{
  if (layers_.size() == 0)
    throw std::runtime_error ("cannot feed-forward an empty net");

  // Calculate forward activation
  layers_[0].forward (x, z[0], a[0]);

  // Aggregate activations for average calculation
  cpu_add (a[0], a_sum_[0], a_sum_[0]);

  for (uint64_t l = 1; l < layers_.size(); l++) {
    // Calculate forward activation
    layers_[l].forward (a[l - 1], z[l], a[l]);

    // Aggregate activations for average calculation except the output layer
    if (l != layers_.size() - 1)
      cpu_add (a[l], a_sum_[l], a_sum_[l]);
  }

  return a[layers_.size() - 1];
}

template<class DataType>
inline DataType Net<DataType>::batchForward (const MatrixType& X)
{
  if (X.height() > batchSize_)
    throw std::runtime_error ("given batch index exceeds batch capacity");

  DataType error = 0;

  for (uint64_t b = 0; b < X.height(); b++) {
    const Vector<DataType> x (X.width(), false, const_cast<DataType*>(X.row(b)));
    auto& y = forward (x, z_[b], a_[b]);

    // Gather L2 norm
    DataType sum = 0;
    // TODO parallelize with OpenMP
    for (uint64_t i = 0; i < x.length(); i++) {
      auto && diff = x[i] - y[i];
      sum += diff * diff;
    }
    error += sum;
  }

  return error;
}

template<class DataType>
inline void Net<DataType>::backward (const VectorType& x, const VectorType& y)
{
  backward (x, y, z_[0], a_[0]);
}

template<class DataType>
inline void Net<DataType>::backward (const VectorType& x, const VectorType& y,
    std::vector<VectorType>& z, std::vector<VectorType>& a)
{
  if (layers_.size() == 0)
    throw std::runtime_error ("cannot back-propagate an empty net");

  // Calculate delta for the output layer
  uint64_t outputLayerId = layers_.size() - 1;
  VectorType &a__ = a[outputLayerId];
  VectorType &z__ = z[outputLayerId];
  VectorType &delta = delta_[outputLayerId];

  // TODO parallelize with OpenMP
  for (uint64_t i = 0; i < y.length(); i++) {
    delta[i] = (a__[i] - y[i]) * layers_[outputLayerId].backwardTransform_(z__[i]);
  }

  for (uint64_t l = outputLayerId; l > 0; l--) {
    // Calculate partial derivatives of weight and bias for layer(l)
    cpu_gemm<DataType> (CblasNoTrans, CblasTrans, 1, delta_[l], a[l - 1], 1., dW_[l]);
    cpu_axpy<DataType> (1., delta_[l], db_[l]);

    // Back-propagate to calculate delta(l-1)
    layers_[l].backward (delta_[l], z[l - 1], delta_[l - 1], a_avg_[l - 1]);
  }

  // Calculate partial derivatives of weight and bias for input layer, i.e. layer(0)
  cpu_gemm<DataType> (CblasNoTrans, CblasTrans, 1, delta_[0], x, 1., dW_[0]);
  cpu_axpy<DataType> (1., delta_[0], db_[0]);
}

template<class DataType>
inline void Net<DataType>::batchBackward (const MatrixType& X, const MatrixType& Y)
{
  assert (X.width() == Y.width() && X.height() == Y.height());

  if (Y.width() > batchSize_)
    throw std::runtime_error ("given batch index exceeds batch capacity");

  for (uint64_t b = 0; b < Y.height(); b++) {
    const Vector<DataType> y (Y.width(), false, const_cast<DataType*>(Y.row(b)));
    const Vector<DataType> x (X.width(), false, const_cast<DataType*>(X.row(b)));
    backward (x, y, z_[b], a_[b]);
  }
}

template<class DataType>
inline void Net<DataType>::averageActivation (unsigned m)
{
  // Average activations except for the last layer
  for (uint64_t l = 0; l < layers_.size() - 1; l++) {
    cpu_axpby<DataType> (1. / m, a_sum_[l], 0., a_avg_[l]);
    a_sum_[l].clear();
  }
}

template<class DataType>
inline void Net<DataType>::deltaParameters (unsigned m)
{
  for (uint64_t l = 0; l < layers_.size(); l++) {
    auto& W = layers_[l].weight();
    auto& dW = dW_[l];
    auto& db = db_[l];

    // 1/m * dW + lambda * W
    cpu_axpby<DataType> (lambda_, W, 1. / m, dW);

    // 1/m * db
    cpu_axpby<DataType> (1. / m, db, 0, db);
  }
}

template<class DataType>
inline void Net<DataType>::updateParameters ()
{
  for (uint64_t l = 0; l < layers_.size(); l++) {
    auto& W = layers_[l].weight();
    auto& b = layers_[l].bias();
    auto& dW = dW_[l];
    auto& db = db_[l];

    // W(l) = W(l) - alpha * dW
    cpu_axpby<DataType> (-alpha_, dW, 1., W);

    // b(l) = b(l) - alpha * db
    cpu_axpby<DataType> (-alpha_, db, 1., b);

    dW.clear ();
    db.clear ();
  }
}

template<class DataType>
inline void Net<DataType>::update (uint64_t layerId, const MatrixType& weight, const VectorType& bias)
{
  assert (layerId < layers_.size());
  layers_[layerId].update (weight, bias);
  dW_[layerId].clear();
  db_[layerId].clear();
}

template<class DataType>
inline DataType Net<DataType>::weightCost () const
{
  DataType weightCost = 0;
  for (uint64_t l = 0; l < layers_.size(); l++) {
    auto &layer = layers_[l];
    weightCost += cpu_dot<DataType> (layer.weight(), layer.weight());
  }
  return weightCost;
}

template<class DataType>
inline DataType Net<DataType>::rhoCost () const
{
  DataType rhoCost = 0;
  for (uint64_t l = 0; l < layers_.size() - 1; l++) {
    auto &rho = a_avg_[l];
    for (uint64_t j = 0; j < rho.length(); j++) {
      rhoCost += rho_ * log (rho_ / rho[j]) + (1 - rho_) * log((1 - rho_) / (1 - rho[j]));
    }
  }
  return rhoCost;
}

}

#endif /* NET_H_ */
