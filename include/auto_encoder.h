#ifndef AUTO_ENCODER_H_
#define AUTO_ENCODER_H_

#include <vector>
#include <string>
#include <cstdio>
#include <cassert>
#include <omp.h>
#include "cmdline.h"
#include "net.h"
#include "lbfgs.h"

#ifdef DUMP_MATFILE
#include <mat.h>
#endif

namespace lightdnn
{

template <class DataType>
class AutoEncoder
{
  public:
    enum class TrainingMethod
    {
      SGD,
      LBFGS
    };

    AutoEncoder (std::string datapath, const OptMap &options);

    virtual ~AutoEncoder ();

    void initialize ();

    void train ();

    void dumpWeights (const char *filename);

  protected:
    void trainSGD ();

    void trainLBFGS ();

    static DataType costFunction (void *encoder, const DataType *parameters,
        DataType *gradients, const int n, const DataType step);

    static int printProgress (void *encoder, const DataType *theta, const DataType *grad,
        const DataType cost, const DataType normTheta, const DataType normGrad, const DataType step,
        int nparam, int niter, int ls);

    std::string datapath;     // path to the data set
    TrainingMethod method;    // training method
    DataType alpha = 0.01;    // learning rate
    DataType beta = 3.;       // weight of sparsity penalty
    DataType rho = 0.01;      // sparsity parameter
    DataType lambda = 0.0001; // weight decay parameter
    unsigned K = 100;         // number of iterations
    unsigned B;               // size of mini-batch
    unsigned N;               // number of training samples
    unsigned P = 1;           // number of processors
    std::vector<unsigned> S;  // layer sizes

    Net<DataType> *net_;
    Matrix<DataType> *training_;
    DataType *parameters_;
    uint64_t numParams_;
};

template<class DataType>
inline AutoEncoder<DataType>::AutoEncoder (std::string path, const OptMap& options) :
  datapath (path), parameters_ (nullptr), numParams_ (0)
{
  method = (options.find ("-method") != options.end ()) ?
      (options.find("-method")->second == "SGD" ? TrainingMethod::SGD : TrainingMethod::LBFGS) :
      TrainingMethod::LBFGS;

  alpha = (options.find ("-alpha") != options.end ()) ?
      atof (options.find ("-alpha")->second.c_str()) : alpha;

  beta = (options.find ("-beta") != options.end ()) ?
      atof (options.find ("-beta")->second.c_str()) : beta;

  rho = (options.find ("-rho") != options.end ()) ?
      atof (options.find ("-rho")->second.c_str()) : rho;

  lambda = (options.find ("-lambda") != options.end ()) ?
      atof (options.find ("-lambda")->second.c_str()) : lambda;

  K = (options.find ("-K") != options.end ()) ?
      atoi (options.find ("-K")->second.c_str()) : K;

  N = (options.find ("-N") != options.end ()) ?
      atoi (options.find ("-N")->second.c_str()) :
      throw std::runtime_error ("number of training samples unspecified");

  B = (options.find ("-B") != options.end ()) ?
      atoi (options.find ("-B")->second.c_str()) : N;

  if (method == TrainingMethod::LBFGS)
    B = N;

  S = (options.find ("-S") != options.end ()) ?
      string2vector (options.find ("-S")->second.c_str()) :
      throw std::runtime_error ("layer sizes unspecified");

  P = (options.find ("-P") != options.end ()) ?
      atoi (options.find ("-P")->second.c_str()) : P;

  printf ("Hyper-parameters:\n"
          "       -N:      %d\n"
          "       -S:      %s\n"
          "       -method: %s\n"
          "       -alpha:  %f\n"
          "       -beta:   %f\n"
          "       -rho:    %f\n"
          "       -lambda: %f\n"
          "       -K:      %d\n"
          "       -B:      %d\n"
          "       -P:      %d\n",
          N, vector2string(S).c_str(),
          (method == TrainingMethod::SGD) ? "SGD" : "LBFGS",
          alpha, beta, rho, lambda, K, B, P);

  net_ = new Net<DataType> ({B, alpha, beta, rho, lambda});

  training_ = new Matrix<DataType> (N, S[0]);
}

template<class DataType>
inline AutoEncoder<DataType>::~AutoEncoder ()
{
  delete net_;
  delete training_;
  delete parameters_;
}

template<class DataType>
inline void AutoEncoder<DataType>::initialize ()
{
  // Set number of cores to be used in OpenMP
  omp_set_num_threads (P);

  // Loading training data
  FILE *file = fopen (datapath.c_str(), "r");
  unsigned numRead = fread (training_->data(), sizeof (DataType), N * S[0], file);
  assert (numRead == N * S[0]);
  fclose (file);

  assert (S.size() > 0);

  // Calculate the total size of model parameters
  for (uint64_t i = 1; i < S.size(); i++) {
    numParams_ += S[i - 1] * S[i] + S[i];
  }

  // Pre-allocate memory for parameters so that it can be shared within L-BFGS solver
  parameters_ = (DataType *) lbfgs_malloc (numParams_);
  DataType *ptr = parameters_;

  // Add layers to the network
  for (uint64_t i = 1; i < S.size(); i++) {
    uint64_t &&height = S[i];
    uint64_t &&width = S[i - 1];
    DataType *weight = ptr;
    DataType *bias = ptr + height * width;
    net_->addLayer ({height, width, MathFunction<DataType>::Type::SIGMOID,
      beta, rho, weight, bias});
    ptr +=  height * width + height;
  }
}

template<class DataType>
inline void AutoEncoder<DataType>::train ()
{
  time_t begin = time (nullptr);
  switch (method) {
    case TrainingMethod::SGD:
      trainSGD ();
      break;
    case TrainingMethod::LBFGS:
      trainLBFGS ();
      break;
  }
  time_t end = time (nullptr);
  printf<GREEN> ("Training time: %ld seconds\n", end - begin);
}

template<class DataType>
inline void AutoEncoder<DataType>::trainSGD ()
{
  printf<GREEN> ("Train the network with Stochastic Gradient Descent...\n");

  // Train the network for a given iteration
  for (unsigned k = 0; k < K; k++) {
    DataType error = 0;
    DataType rhoCost = 0;
    unsigned numBatches = N / B + (N % B != 0);
    // For each mini-batch
    for (unsigned b = 0; b < numBatches; b++) {
      // Assemble a mini-batch
      unsigned m = ((b + 1) * B > N) ? (N - b * B) : B;
      Matrix<DataType> samples (m, training_->width(), false, training_->row(b * B));

      // Feed-forward a mini-batch
      error += net_->batchForward (samples);

      // Update average activations
      net_->averageActivation (m);

      // Back-propagate a mini-batch
      net_->batchBackward (samples, samples);

      // Calculate delta weight/bias
      net_->deltaParameters (m);

      // Update parameters
      net_->updateParameters ();

      // Calculate the cost
      rhoCost += net_->rhoCost ();
    }
    DataType weightCost = net_->weightCost ();
    DataType cost = 0.5 * error / N + 0.5 * lambda * weightCost + beta * rhoCost / numBatches;
    printf<INFO> ("\rIteration %u/%u (cost = %.5f)        ", k + 1, K, (float) cost);
  }
  printf ("\n");
}

template<class DataType>
inline void AutoEncoder<DataType>::trainLBFGS ()
{
  printf<GREEN> ("Train the network with L-BFGS solver...\n");

  DataType cost;
  lbfgs_parameter_t lbfgsParam;
  lbfgs_parameter_init (&lbfgsParam);
  lbfgsParam.max_iterations = K;
  int ret = lbfgs (numParams_, parameters_, &cost, costFunction, printProgress, this, &lbfgsParam);
  printf<GREEN> ("\nL-BFGS solver terminated with status code %d\n", ret);
}

template<class DataType>
inline DataType AutoEncoder<DataType>::costFunction (void* instance, const DataType* parameters,
    DataType* gradients, const int n, const DataType step)
{
  AutoEncoder<DataType> *encoder = (AutoEncoder<DataType>*) instance;
  Net<DataType> *net = encoder->net_;
  Matrix<DataType> &X = *encoder->training_;
  unsigned m = X.height();

  // Feed-forward
  DataType error = net->batchForward (X);

  // Update average activations
  net->averageActivation (m);

  // Back-propagate
  net->batchBackward (X, X);

  // Calculate delta weight/bias
  net->deltaParameters (m);

  // Copy dW/db to the gradient space in l-bfgs
  DataType *ptr = gradients;
  for (uint64_t l = 0; l < net->numLayers(); l++) {
    auto &dW = net->dW (l);
    auto &db = net->db (l);
    dW.copyTo (ptr);
    ptr += dW.height() * dW.width();
    db.copyTo (ptr);
    ptr += db.length();
  }

  // Calculate the cost
  DataType weightCost = net->weightCost ();
  DataType rhoCost = net->rhoCost ();
  DataType cost = 0.5 * error / m + 0.5 * encoder->lambda * weightCost + encoder->beta * rhoCost;
  return cost;
}

template<class DataType>
inline int AutoEncoder<DataType>::printProgress (void* instance, const DataType* theta,
    const DataType *grad, const DataType cost, const DataType normTheta,
    const DataType normGrad, const DataType step, int nparam, int niter, int ls)
{
  AutoEncoder<DataType> *encoder = (AutoEncoder<DataType>*) instance;
  printf<INFO> ("\rIteration %d/%d (cost = %0.5f)        ", niter, encoder->K, cost);
  return 0;
}

#ifdef DUMP_MATFILE
template<class DataType>
mxArray* createNumericMatrix (uint64_t height, uint64_t width);

template<>
inline mxArray* createNumericMatrix<float> (uint64_t height, uint64_t width)
{
  return mxCreateNumericMatrix (height, width, mxSINGLE_CLASS, mxREAL);
}

template<>
inline mxArray* createNumericMatrix<double> (uint64_t height, uint64_t width)
{
  return mxCreateNumericMatrix (height, width, mxDOUBLE_CLASS, mxREAL);
}

#endif

template<class DataType>
inline void AutoEncoder<DataType>::dumpWeights (const char *filename)
{
#ifdef DUMP_MATFILE
  MATFile *matfile = matOpen (filename, "w");
  if (matfile == nullptr) {
    printf ("error: failed to create file %s\n", filename);
    exit (1);
  }

  for (size_t i = 0; i < S.size() - 1; i++) {
    auto &weight = net_->layer (i).weight();
    weight.transpose ();
    mxArray *weightArray = createNumericMatrix<DataType> (weight.width(), weight.height());
    memcpy (mxGetPr (weightArray), weight.data(),
        weight.height() * weight.width() * sizeof(DataType));
    char name[16];
    sprintf (name, "W%zu", i);
    int status = matPutVariable (matfile, name, weightArray);
    if (status != 0) {
      printf ("error: failed to dump weight matrix %zu to file %s\n", i, filename);
      exit (1);
    }
  }

  if (matClose (matfile) != 0) {
    printf ("error: failed to close file %s\n", filename);
    exit (1);
  }
#else
  for (size_t i = 0; i < S.size() - 1; i++) {
    auto &weight = net_->layer (i).weight();
    char filename[20];
    sprintf (filename, "W%zu.bin", i);
    FILE *file = fopen (filename, "w");
    fwrite (weight.data(), sizeof(DataType), weight.height() * weight.width(), file);
    fclose (file);
  }
#endif
}

}

#endif /* AUTO_ENCODER_H_ */
