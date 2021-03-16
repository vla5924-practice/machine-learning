#pragma once

#include <cassert>
#include <cmath>

#include "maths/types.hpp"

namespace Algorithms {
namespace Maths {

template <typename T>
Matrix<T> createMatrix(size_t rows, size_t cols, T fill = T{}) {
    Matrix<T> matrix(rows);
    for (size_t i = 0; i < rows; i++)
        matrix[i].resize(cols, fill);
    return matrix;
}

template <typename T>
Matrix<T> createIdentity(size_t dim) {
    Matrix<T> matrix(dim);
    for (size_t i = 0; i < dim; i++) {
        matrix[i].resize(dim, 0);
        matrix[i][i] = 1;
    }
    return matrix;
}

template <typename T>
T dot(const Vector<T>& x, const Vector<T>& y) {
    assert(x.size() == y.size());
    T result{};
    size_t size = x.size();
    for (size_t i = 0; i < size; i++)
        result += x[i] * y[i];
    return result;
}

template <typename T>
T norm(const Vector<T>& x) {
    T sum{};
    for (const T& elem : x)
        sum += elem * elem;
    return std::sqrt(sum);
}

template <typename T>
size_t nrows(const Matrix<T>& x) {
    return x.size();
}

template <typename T>
size_t ncols(const Matrix<T>& x) {
    return x.size() == 0 ? 0 : x.front().size();
}

template <typename T>
std::pair<size_t, size_t> ndims(const Matrix<T>& x) {
    return std::make_pair(nrows(x), ncols(x));
}

} // namespace Maths
} // namespace Algorithms
