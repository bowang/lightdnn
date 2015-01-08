#ifndef MATH_FUNCTIONS_H_
#define MATH_FUNCTIONS_H_

#include <cmath>
#include <random>
#include <stdexcept>

namespace lightdnn
{

template<class DataType>
inline DataType identity (const DataType x)
{
  return x;
}

template<class DataType>
inline DataType d_identity (const DataType x)
{
  return 1.;
}

template<class DataType>
inline DataType sigmoid (const DataType x)
{
  return 1. / (1. + exp (-x));
}

template<class DataType>
inline DataType d_sigmoid (const DataType x)
{
  DataType sigm = sigmoid<DataType> (x);
  return sigm * (1. - sigm);
}

template<class DataType>
class MathFunction
{
  public:
    typedef DataType (*MathFunc) (const DataType);

    enum class Type
    {
      IDENTITY,
      SIGMOID
    };

    static MathFunc function (Type type);

    static MathFunc deriviative (Type type);

    static const unsigned int rand_seed = 2015;

    static std::mt19937 rand_engine;
};

template<class DataType>
inline typename MathFunction<DataType>::MathFunc
MathFunction<DataType>::function (MathFunction<DataType>::Type type)
{
  switch (type) {
    case Type::IDENTITY:
      return identity<DataType>;
    case Type::SIGMOID:
      return sigmoid<DataType>;
    default:
      throw std::runtime_error ("unknown function type");
  }
}

template<class DataType>
inline typename MathFunction<DataType>::MathFunc
MathFunction<DataType>::deriviative (MathFunction<DataType>::Type type)
{
  switch (type) {
    case Type::IDENTITY:
      return d_identity<DataType>;
    case Type::SIGMOID:
      return d_sigmoid<DataType>;
    default:
      throw std::runtime_error ("unknown function type");
  }
}

}

#endif /* MATH_FUNCTIONS_H_ */
