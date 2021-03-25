#pragma once

#include <tuple>

#include "maths/functions.hpp"
#include "maths/types.hpp"

namespace Algorithms {
namespace Maths {

template <typename T>
Vector<T> operator*(const T& x, const Vector<T>& y) {
    Vector<T> result = y;
    for (auto& elem : result)
        elem = x * elem;
    return result;
}

template <typename T>
Matrix<T> operator*(const T& x, const Matrix<T>& y) {
    Matrix<T> result = y;
    for (auto& row : result)
        for (auto& elem : row)
            elem = x * elem;
    return result;
}

template <typename T>
Vector<T> operator+(const Vector<T>& x, const Vector<T>& y) {
    assert(x.size() == y.size());
    size_t size = x.size();

    Vector<T> result(size);
    for (size_t i = 0; i < size; i++)
        result[i] = x[i] + y[i];
    return result;
}

template <typename T>
Vector<T> operator-(const Vector<T>& x, const Vector<T>& y) {
    assert(x.size() == y.size());
    size_t size = x.size();

    Vector<T> result(size);
    for (size_t i = 0; i < size; i++)
        result[i] = x[i] - y[i];
    return result;
}

template <typename T>
Matrix<T> operator*(const Vector<T>& x, const Vector<T>& y) {
    size_t rows = x.size();
    size_t cols = y.size();

    Matrix<T> result = createMatrix<T>(rows, cols);
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            result[i][j] = x[i] * y[j];
    return result;
}

template <typename T>
Vector<T> operator*(const Matrix<T>& x, const Vector<T>& y) {
    size_t rows, cols;
    std::tie(rows, cols) = ndims(x);
    assert(cols == y.size());

    Vector<T> result(rows, T{});
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            result[i] += x[i][j] * y[j];
    return result;
}

template <typename T>
Vector<T> operator*(const Vector<T>& x, const Matrix<T>& y) {
    size_t rows, cols;
    std::tie(rows, cols) = ndims(y);

    Vector<T> result(cols, T{});
    for (size_t i = 0; i < cols; i++)
        for (size_t j = 0; j < rows; j++)
            result[i] += x[j] * y[j][i];
    return result;
}

template <typename T>
Matrix<T> operator-(const Matrix<T>& x) {
    Matrix<T> result = x;
    for (auto& row : result)
        for (auto& elem : row)
            elem = -elem;
    return result;
}

template <typename T>
Vector<T> operator-(const Vector<T>& x) {
    Vector<T> result = x;
    for (auto& elem : result)
        elem = -elem;
    return result;
}

template <typename T>
Matrix<T> operator*(const Matrix<T>& x, const Matrix<T>& y) {
    size_t x_cols = ncols(x);
    assert(x_cols == nrows(y));
    size_t rows = nrows(x);
    size_t cols = ncols(y);
    Matrix<T> result = createMatrix<T>(rows, cols);
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            for (size_t k = 0; k < x_cols; k++)
                result[i][j] += x[i][k] * y[k][j];
    return result;
}

} // namespace Maths
} // namespace Algorithms
