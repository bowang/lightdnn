#ifndef BASIC_TYPES_H_
#define BASIC_TYPES_H_

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cassert>

namespace lightdnn
{

template<class DataType>
class Matrix
{
  public:
    /*
     * @brief constructor
     */
    Matrix (uint64_t height, uint64_t width, bool clear = false, DataType *data = nullptr);

    /*
     * @brief move constructor
     */
    Matrix (Matrix&&);

    /*
     * @brief move operator
     */
    Matrix& operator = (Matrix&&);

    /*
     * @brief copy constructor (deleted)
     */
    Matrix (const Matrix&) = delete;

    /*
     * @brief copy operator
     */
    Matrix& operator = (const Matrix&);

    virtual ~Matrix ();

    uint64_t height () const { return height_; }

    uint64_t width () const { return width_; }

    DataType* data () const { return data_; }

    DataType* col (uint64_t colId) const;

    DataType* row (uint64_t rowId) const;

    void transpose ();

    void clear ();

    void copyTo (DataType *dst) const;

    DataType& operator [](uint64_t i) const { return this->data_[i]; }

    void print (bool transpose = false) const;

  protected:
    uint64_t height_;
    uint64_t width_;
    bool allocated_;
    DataType* data_;
};

template<class DataType>
inline Matrix<DataType>::Matrix (uint64_t height, uint64_t width, bool clear, DataType *data) :
    height_ (height), width_ (width), allocated_ (data == nullptr), data_ (data)
{
  if (data_ == nullptr) {
    if (clear)
      data_ = (DataType*) calloc (height * width, sizeof(DataType));
    else
      data_ = (DataType*) malloc (height * width * sizeof(DataType));
  }
  else {
    if (clear)
      memset (data_, 0, height * width * sizeof(DataType));
  }
}

template<class DataType>
inline Matrix<DataType>::Matrix (Matrix&& that) :
    height_ (that.height_), width_ (that.width_),
    allocated_ (that.allocated_), data_ (that.data_)
{
  that.allocated_ = false;
  that.data_ = nullptr;
}

template<class DataType>
inline Matrix<DataType>::~Matrix ()
{
  if (allocated_)
    free (data_);
}

template<class DataType>
inline Matrix<DataType>& Matrix<DataType>::operator = (Matrix&& that)
{
  assert (this->height_ == that.height_);
  assert (this->width_ == that.width_);
  this->data_ = that.data_;
  that.data_ = nullptr;
}

template<class DataType>
inline Matrix<DataType>& Matrix<DataType>::operator = (const Matrix& that)
{
  assert (this->height_ == that.height_);
  assert (this->width_ == that.width_);
  memcpy (this->data_, that.data_, height_ * width_);
}

template<class DataType>
inline DataType* Matrix<DataType>::col (uint64_t colId) const
{
  throw std::runtime_error("call to unimplemented function col()");
}

template<class DataType>
inline DataType* Matrix<DataType>::row (uint64_t rowId) const
{
  return &data_[rowId * width_];
}

template<class DataType>
inline void Matrix<DataType>::transpose ()
{
  const uint64_t count = height_ * width_;
  const uint64_t count1 = height_ * width_ - 1;
  std::vector<bool> visited (count);
  uint64_t i = 0;
  while (++i != count) {
    if (visited[i]) continue;
    uint64_t j = i;
    do {
      j = (j == count1) ? count1 : (height_ * j) % count1;
      std::swap (data_[i], data_[j]);
      visited[j] = true;
    } while (j != i);
  }
  std::swap (height_, width_);
}

template<class DataType>
inline void Matrix<DataType>::clear ()
{
  memset (data_, 0, height_ * width_ * sizeof(DataType));
}

template<class DataType>
inline void Matrix<DataType>::copyTo (DataType* dst) const
{
  memcpy (dst, data_, height_ * width_ * sizeof(DataType));
}

template<class DataType>
inline void Matrix<DataType>::print (bool transpose) const
{
  // Print a matrix of m rows, n cols
  uint64_t m = transpose ? width_ : height_;
  uint64_t n = transpose ? height_ : width_;

  if (!transpose) {
    for (uint64_t i = 0; i < m; i++) {
      for (uint64_t j = 0; j < n; j++)
        std::cout << std::fixed << std::setprecision(4) << data_[i * n + j] << " ";
      std::cout << std::endl;
    }
  }
  else {
    for (uint64_t i = 0; i < m; i++) {
      for (uint64_t j = 0; j < n; j++)
        std::cout << std::fixed << std::setprecision(4) << data_[(m - i - 1) + j * m] << " ";
      std::cout << std::endl;
    }
  }

  std::cout << std::endl;
}

template<class DataType>
class Vector : public Matrix<DataType>
{
  public:
    /*
     * @brief constructor
     */
    Vector (uint64_t length, bool clear = false, DataType *data = nullptr) :
      Matrix<DataType> (length, 1, clear, data) {}

    /*
     * @brief move operator
     */
    Vector (Vector&&) = default;

    /*
     * @brief move operator
     */
    Vector& operator = (Vector&&) = default;

    /*
     * @brief copy constructor (deleted)
     */
    Vector (const Vector&) = delete;

    /*
     * @brief copy operator
     */
    Vector& operator = (const Vector&) = default;

    virtual ~Vector () {}

    uint64_t length () const { return this->height_; }
};

}

#endif /* BASIC_TYPES_H_ */
