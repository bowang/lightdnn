#include "math_functions.h"

namespace lightdnn
{

template<>
std::mt19937 MathFunction<float>::rand_engine (MathFunction<float>::rand_seed);

template<>
std::mt19937 MathFunction<double>::rand_engine (MathFunction<double>::rand_seed);

}
