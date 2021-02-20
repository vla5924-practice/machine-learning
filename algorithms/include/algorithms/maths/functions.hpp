#pragma once

#include <cassert>
#include <cmath>

#include "maths/types.hpp"

namespace Algorithms
{
namespace Maths
{

template<typename T>
Matrix<T> createMatrix(size_t rows, size_t cols, T fill = T{})
{
    Matrix<T> matrix(rows);
    for (size_t i = 0; i < rows; i++)
        matrix[i].resize(cols, fill);
    return matrix;
}

template<typename T>
Matrix<T> createIdentity(size_t dim)
{
    Matrix<T> matrix(dim);
    for (size_t i = 0; i < dim; i++)
    {
        matrix[i].resize(dim, 0);
        matrix[i][i] = 1;
    }
    return matrix;
}

template<typename T>
T dot(const Vector<T>& x, const Vector<T>& y)
{
    assert(x.size() == y.size());
    T result{};
    size_t size = x.size();
    for (size_t i = 0; i < size; i++)
        result += x[i] * y[i];
    return result;
}

template<typename T>
T norm(const Vector<T>& x)
{
    T sum = T{};
    for (const T& elem : x)
        sum += elem * elem;
    return std::sqrt(sum);
}

} // namespace Maths
} // namespace Algorithms
