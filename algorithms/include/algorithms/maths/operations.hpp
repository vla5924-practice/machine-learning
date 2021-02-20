#pragma once

#include "maths/functions.hpp"
#include "maths/types.hpp"

namespace Algorithms
{
namespace Maths
{

template<typename T>
Matrix<T> operator*(const Vector<T>& x, const Vector<T>& y)
{
    size_t rows = x.size();
    size_t cols = y.size();
    Matrix<T> result = createZeros(rows, cols);
    for (size_t i = 0; i < cols; i++)
        for (size_t j = 0; j < rows; j++)
            result[i] = x[i] * y[j];
    return result;
}

template<typename T>
Vector<T> operator*(const Matrix<T>& x, const Vector<T>& y)
{
    size_t rows = x.size();
    size_t cols = x.front().size();
    Vector<T> result(rows, T{});
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            result[i] += x[i][j] * y[j];
    return result;
}

template<typename T>
Vector<T> operator*(const Vector<T>& x, const Matrix<T>& y)
{
    size_t rows = y.size();
    size_t cols = y.front().size();
    Vector<T> result(cols, T{});
    for (size_t i = 0; i < cols; i++)
        for (size_t j = 0; j < rows; j++)
            result[i] += x[j] * y[j][i]; // TODO: double check this
    return result;
}

} // namespace Maths
} // namespace Algorithms
