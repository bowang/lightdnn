#ifndef LAYER_H_
#define LAYER_H_

#include "basic_types.h"
#include "math_functions.h"
#include "blas_routines.h"

namespace lightdnn
{

template<class DataType>
class Net;

template<class DataType>
class Layer
{
  public:
    typedef Vector<DataType> VectorType;
    typedef Matrix<DataType> MatrixType;
    typedef typename MathFunction<DataType>::MathFunc MathFuncType;

    struct LayerParameter
    {
      uint64_t height;
      uint64_t width;
      typename MathFunction<DataType>::Type type;
      DataType beta;
      DataType rho;
      DataType *weight;
      DataType *bias;
    };

    Layer (const LayerParameter& param);

    Layer (Layer&&);

    Layer& operator = (Layer&&);

    Layer (const Layer&) = delete;

    Layer& operator = (const Layer&) = delete;

    virtual ~Layer ();

    void forward (const VectorType& x, VectorType& z, VectorType& a);

    void backward (const VectorType& deltaIn, const VectorType& z, VectorType& deltaOut);

    void backward (const VectorType& deltaIn, const VectorType& z, VectorType& deltaOut, const VectorType& rho);

    void update (const MatrixType& weight, const VectorType& bias);

    MatrixType& weight () const { return *weight_; }

    VectorType& bias () const { return *bias_; }

    uint64_t size () const { return bias_->length (); }

    friend class Net<DataType>;

  protected:
    DataType beta_;
    DataType rho_;
    MathFuncType forwardTransform_;
    MathFuncType backwardTransform_;
    MatrixType* weight_;
    VectorType* bias_;
};

template<class DataType>
inline Layer<DataType>::Layer (const LayerParameter& param) :
    beta_ (param.beta), rho_ (param.rho),
    forwardTransform_ (MathFunction<DataType>::function (param.type)),
    backwardTransform_ (MathFunction<DataType>::deriviative (param.type))
{
  weight_ = new MatrixType (param.height, param.width, false, param.weight);
  bias_ = new VectorType (param.height, true, param.bias);

  // Initialize weights under Gaussian distribution
  std::normal_distribution<> normal_dist (0, sqrt (6. / DataType(param.height + param.width + 1)));

  DataType *W = weight_->data();
  for (uint64_t i = 0; i < param.height * param.width; i++) {
    W[i] = normal_dist (MathFunction<DataType>::rand_engine);
  }
}

template<class DataType>
inline Layer<DataType>::Layer (Layer&& that)
{
  *this = std::move (that);
}

template<class DataType>
inline Layer<DataType>& Layer<DataType>::operator = (Layer&& that)
{
  this->weight_ = that.weight_;
  this->bias_ = that.bias_;
  this->beta_ = that.beta_;
  this->rho_ = that.rho_;
  this->forwardTransform_ = that.forwardTransform_;
  this->backwardTransform_ = that.backwardTransform_;
  that.weight_ = nullptr;
  that.bias_ = nullptr;
  return *this;
}

template<class DataType>
inline Layer<DataType>::~Layer ()
{
  delete weight_;
  delete bias_;
}

template<class DataType>
inline void Layer<DataType>::forward (const VectorType& x, VectorType& z, VectorType& a)
{
  // z1(l+1) = W(l) * a(l)
  cpu_gemv<DataType>(CblasNoTrans, 1., *weight_, x, 0., z);

  // z(l+1) = z1(l+1) + b(l)
  cpu_add<DataType>(*bias_, z, z);

  // a(l+1) = f(z(l+1))
  cpu_fx<DataType>(z, a, forwardTransform_);
}

template<class DataType>
inline void Layer<DataType>::backward (const VectorType& deltaIn, const VectorType& z, VectorType& deltaOut)
{
  assert (deltaOut.length() == z.length());

  // delta1(l) = W(l)^T * delta(l+1)
  cpu_gemv<DataType>(CblasTrans, 1., *weight_, deltaIn, 0., deltaOut);

  // TODO parallelize with OpenMP
  // delta(l) = delta1(l) .* f'(z(l))
  for (uint64_t i = 0; i < z.length(); i++) {
    deltaOut[i] *= backwardTransform_(z[i]);
  }
}

template<class DataType>
inline void Layer<DataType>::backward (const VectorType& deltaIn, const VectorType& z,
    VectorType& deltaOut, const VectorType& rho)
{
  assert (deltaOut.length() == rho.length());
  assert (deltaOut.length() == z.length());

  // delta1(l) = W(l)^T * delta(l+1)
  cpu_gemv<DataType>(CblasTrans, 1., *weight_, deltaIn, 0., deltaOut);

  // TODO parallelize with OpenMP
  // delta(l) = delta1(l) .* f'(z(l))
  for (uint64_t i = 0; i < z.length(); i++) {
    // Add sparsity penalty
    deltaOut[i] += beta_ * ((1. - rho_) / (1. - rho[i]) - rho_ / rho[i]);
    // Multiple by derivative
    deltaOut[i] *= backwardTransform_(z[i]);
  }
}

template<class DataType>
inline void Layer<DataType>::update (const MatrixType& weight, const VectorType& bias)
{
  *weight_ = weight;
  *bias_ = bias;
}

}

#endif /* LAYER_H_ */
