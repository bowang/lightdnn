#ifndef BLAS_ROUTINES_H_
#define BLAS_ROUTINES_H_

#include <cblas.h>
#include "basic_types.h"
#include "debug.h"

namespace lightdnn
{

/*
 * @brief C <-- alpha * A^TransA * B^TransB + beta * C
 */
template<class DataType>
void cpu_gemm (const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const DataType alpha, const Matrix<DataType>& A, const Matrix<DataType>& B,
    const DataType beta, Matrix<DataType>& C);

template<>
inline void cpu_gemm<float> (const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const float alpha, const Matrix<float>& A, const Matrix<float>& B,
    const float beta, Matrix<float>& C)
{
  assert (((TransA == CblasNoTrans) ? A.width() : A.height()) ==
      ((TransB == CblasNoTrans) ? B.height() : B.width() ));
  assert (((TransA == CblasNoTrans) ? A.height() : A.width()) == C.height());
  assert (((TransB == CblasNoTrans) ? B.width() : A.height()) == C.width());

  uint64_t M = (TransA == CblasNoTrans) ? A.height() : A.width();
  uint64_t N = (TransB == CblasNoTrans) ? B.width() : B.height();
  uint64_t K = (TransA == CblasNoTrans) ? A.width() : A.height();
  uint64_t lda = (TransA == CblasNoTrans) ? K : M;
  uint64_t ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm (CblasRowMajor, TransA, TransB, M, N, K,
      alpha, A.data(), lda, B.data(), ldb, beta, C.data(), N);
}

template<>
inline void cpu_gemm<double> (const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const double alpha, const Matrix<double>& A, const Matrix<double>& B,
    const double beta, Matrix<double>& C)
{
  assert (((TransA == CblasNoTrans) ? A.width() : A.height()) ==
      ((TransB == CblasNoTrans) ? B.height() : B.width() ));
  assert (((TransA == CblasNoTrans) ? A.height() : A.width()) == C.height());
  assert (((TransB == CblasNoTrans) ? B.width() : B.height()) == C.width());

  uint64_t M = (TransA == CblasNoTrans) ? A.height() : A.width();
  uint64_t N = (TransB == CblasNoTrans) ? B.width() : B.height();
  uint64_t K = (TransA == CblasNoTrans) ? A.width() : A.height();
  uint64_t lda = (TransA == CblasNoTrans) ? K : M;
  uint64_t ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm (CblasRowMajor, TransA, TransB, M, N, K,
      alpha, A.data(), lda, B.data(), ldb, beta, C.data(), N);
}

/*
 * @brief y <-- alpha * A^TransA * x + beta * y
 */
template<class DataType>
void cpu_gemv (const CBLAS_TRANSPOSE TransA,
    const DataType alpha, const Matrix<DataType>& A, const Vector<DataType>& x,
    const DataType beta, Vector<DataType>& y);

template<>
inline void cpu_gemv<float> (const CBLAS_TRANSPOSE TransA,
    const float alpha, const Matrix<float>& A, const Vector<float>& x,
    const float beta, Vector<float>& y)
{
  assert (((TransA == CblasNoTrans) ? A.width() : A.height()) == x.length());
  assert (((TransA == CblasNoTrans) ? A.height() : A.width()) == y.length());

  uint64_t M = A.height();
  uint64_t N = A.width();
  cblas_sgemv (CblasRowMajor, TransA, M, N,
      alpha, A.data(), N, x.data(), 1, beta, y.data(), 1);
}

template<>
inline void cpu_gemv<double> (const CBLAS_TRANSPOSE TransA,
    const double alpha, const Matrix<double>& A, const Vector<double>& x,
    const double beta, Vector<double>& y)
{
  assert (((TransA == CblasNoTrans) ? A.width() : A.height()) == x.length());
  assert (((TransA == CblasNoTrans) ? A.height() : A.width()) == y.length());

  uint64_t M = A.height();
  uint64_t N = A.width();
  cblas_dgemv (CblasRowMajor, TransA, M, N,
      alpha, A.data(), N, x.data(), 1, beta, y.data(), 1);
}

/*
 * @brief y <-- x
 */
template<class DataType>
void cpu_copy (const Matrix<DataType>& x, Matrix<DataType>& y);

template<>
inline void cpu_copy<float> (const Matrix<float>& x, Matrix<float>& y)
{
  assert (x.height() == y.height());
  assert (x.width() == y.width());
  cblas_scopy (x.height() * x.width(), x.data(), 1, y.data(), 1);
}

template<>
inline void cpu_copy<double> (const Matrix<double>& x, Matrix<double>& y)
{
  assert (x.height() == y.height());
  assert (x.width() == y.width());
  cblas_dcopy (x.height() * x.width(), x.data(), 1, y.data(), 1);
}

/*
 * @brief y <-- alpha * x + y
 */
template<class DataType>
void cpu_axpy (const DataType alpha, const Matrix<DataType>& x, Matrix<DataType>& y);

template<>
inline void cpu_axpy<float> (const float alpha, const Matrix<float>& x, Matrix<float>& y)
{
  assert (x.width() == y.width());
  assert (x.height() == y.height());
  cblas_saxpy (x.height() * x.width(), alpha, x.data(), 1, y.data(), 1);
}

template<>
inline void cpu_axpy<double> (const double alpha, const Matrix<double>& x, Matrix<double>& y)
{
  assert (x.width() == y.width());
  assert (x.height() == y.height());
  cblas_daxpy (x.height() * x.width(), alpha, x.data(), 1, y.data(), 1);
}

/*
 * @brief y <-- alpha * x + beta * y
 */
template<class DataType>
void cpu_axpby (const DataType alpha, const Matrix<DataType>& x, const DataType beta, Matrix<DataType>& y);

template<>
inline void cpu_axpby<float> (const float alpha, const Matrix<float>& x, const float beta, Matrix<float>& y)
{
  assert (x.width() == y.width());
  assert (x.height() == y.height());
  cblas_saxpby (x.height() * x.width(), alpha, x.data(), 1, beta, y.data(), 1);
}

template<>
inline void cpu_axpby<double> (const double alpha, const Matrix<double>& x, const double beta, Matrix<double>& y)
{
  assert (x.width() == y.width());
  assert (x.height() == y.height());
  cblas_daxpby (x.height() * x.width(), alpha, x.data(), 1, beta, y.data(), 1);
}

/*
 * @brief returns x' * y
 */
template<class DataType>
DataType cpu_dot (const Matrix<DataType>& x, const Matrix<DataType>& y);

template<>
inline float cpu_dot<float> (const Matrix<float>& x, const Matrix<float>& y)
{
  assert (x.width() == y.width());
  assert (x.height() == y.height());
  return cblas_sdot (x.height() * x.width(), x.data(), 1, y.data(), 1);
}

template<>
inline double cpu_dot<double> (const Matrix<double>& x, const Matrix<double>& y)
{
  assert (x.width() == y.width());
  assert (x.height() == y.height());
  return cblas_ddot (x.height() * x.width(), x.data(), 1, y.data(), 1);
}

/*
 * @brief returns L2 norm of x, i.e. sqrt (x' * x)
 */
template<class DataType>
DataType cpu_nrm2 (const Vector<DataType>& x);

template<>
inline float cpu_nrm2<float> (const Vector<float>& x)
{
  return cblas_snrm2 (x.length(), x.data(), 1);
}

template<>
inline double cpu_nrm2<double> (const Vector<double>& x)
{
  return cblas_dnrm2 (x.length(), x.data(), 1);
}

// TODO parallelize with OpenMP
/*
 * @brief y <-- f(x)
 */
template<class DataType>
inline void cpu_fx (const Matrix<DataType>& x, Matrix<DataType>& y, DataType (*f)(DataType))
{
  assert (x.width() == y.width());
  assert (x.height() == y.height());
  for (uint64_t i = 0; i < y.height() * y.width(); i++) {
    y[i] = f (x[i]);
  }
}

// TODO accelerate with SSE instructions and loop unrolling
// TODO parallelize with OpenMP
/*
 * @brief y <-- f(a, b)
 */
#define DEFINE_VSL_BINARY_FUNC(name, operation) \
  template<class DataType> \
  inline void cpu_##name (const Matrix<DataType>& a, const Matrix<DataType>& b, Matrix<DataType>& y) \
  { \
    assert (a.width() == b.width()); \
    assert (a.height() == b.height()); \
    assert (a.width() == y.width()); \
    assert (a.height() == y.height()); \
    for (uint64_t i = 0; i < y.height() * y.width(); i++) { operation; } \
  } \

DEFINE_VSL_BINARY_FUNC(add, y[i] = a[i] + b[i]);
DEFINE_VSL_BINARY_FUNC(sub, y[i] = a[i] - b[i]);
DEFINE_VSL_BINARY_FUNC(mul, y[i] = a[i] * b[i]);
DEFINE_VSL_BINARY_FUNC(div, y[i] = a[i] / b[i]);

}

#endif /* BLAS_ROUTINES_H_ */
