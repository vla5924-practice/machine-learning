#include <gtest/gtest.h>

#include <algorithms/maths/operations.hpp>

using namespace Algorithms::Maths;

TEST(Algorithms_Math_Operations__Test, can_multiply_number_by_vector)
{
    Vector<int> vec = { 1, 3, 5 };
    Vector<int> act = 2 * vec;
    Vector<int> exp = { 2, 6, 10 };
    ASSERT_EQ(act, exp);
}
