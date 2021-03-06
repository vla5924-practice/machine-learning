#pragma once

#include <vector>

namespace Algorithms {
namespace Maths {

template <typename T>
using Vector = std::vector<T>;

template <typename T>
using Matrix = Vector<Vector<T>>;

} // namespace Maths
} // namespace Algorithms
