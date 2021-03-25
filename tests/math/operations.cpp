#include <gtest/gtest.h>

#include <algorithms/maths/operations.hpp>

using namespace Algorithms::Maths;

TEST(Algorithms_Math_Operations__Test, can_multiply_number_by_vector) {
    Vector<int> vec = {1, 3, 5};
    Vector<int> act = 2 * vec;
    Vector<int> exp = {2, 6, 10};
    ASSERT_EQ(act, exp);
}

TEST(Algorithms_Math_Operations__Test, can_multiply_number_by_matrix) {
    Matrix<int> mat = {{1, 3, 5}, {2, 4, 6}};
    Matrix<int> act = 2 * mat;
    Matrix<int> exp = {{2, 6, 10}, {4, 8, 12}};
    ASSERT_EQ(act, exp);
}

TEST(Algorithms_Math_Operations__Test, can_add_vectors_with_equal_sizes) {
    Vector<int> vec1 = {1, 3, 5};
    Vector<int> vec2 = {2, 4, 6};
    Vector<int> act = vec1 + vec2;
    Vector<int> exp = {3, 7, 11};
    ASSERT_EQ(act, exp);
}

TEST(Algorithms_Math_Operations__Test, cannot_add_vectors_with_different_sizes) {
    Vector<int> vec1 = {1, 3, 5};
    Vector<int> vec2 = {2, 4};
    ASSERT_DEATH(vec1 + vec2, ".*");
}

TEST(Algorithms_Math_Operations__Test, can_sub_vector_from_vector) {
    Vector<int> vec1 = {1, 3, 5};
    Vector<int> vec2 = {2, 4, 6};
    Vector<int> act = vec1 - vec2;
    Vector<int> exp = {-1, -1, -1};
    ASSERT_EQ(act, exp);
}

TEST(Algorithms_Math_Operations__Test, cannot_sub_vectors_with_different_sizes) {
    Vector<int> vec1 = {1, 3, 5};
    Vector<int> vec2 = {2, 4};
    ASSERT_DEATH(vec1 - vec2, ".*");
}

TEST(Algorithms_Math_Operations__Test, can_multiply_vector_by_vector) {
    Vector<int> vec1 = {1, 3, 5};
    Vector<int> vec2 = {2, 4, 6};
    Matrix<int> act = vec1 * vec2;
    Matrix<int> exp = {{2, 4, 6}, {6, 12, 18}, {10, 20, 30}};
    ASSERT_EQ(act, exp);
}

TEST(Algorithms_Math_Operations__Test, can_multiply_vector_by_vector_with_different_sizes) {
    Vector<int> vec1 = {1, 3, 5};
    Vector<int> vec2 = {2, 4};
    Matrix<int> act = vec1 * vec2;
    Matrix<int> exp = {{2, 4}, {6, 12}, {10, 20}};
    ASSERT_EQ(act, exp);
}

TEST(Algorithms_Math_Operations__Test, can_multiply_matrix_and_vector_with_valid_sizes) {
    Matrix<int> mat = {{1, 3, 5}, {2, 4, 6}};
    Vector<int> vec = {-1, 0, 1};
    Vector<int> act = mat * vec;
    Vector<int> exp = {4, 4};
    ASSERT_EQ(act, exp);
}

TEST(Algorithms_Math_Operations__Test, can_multiply_matrix_and_vector_with_equal_sizes) {
    Matrix<int> mat = {{1, 3, 5}, {2, 4, 6}, {-1, 0, 1}};
    Vector<int> vec = {13, 17, 15};
    Vector<int> act = mat * vec;
    Vector<int> exp = {139, 184, 2};
    ASSERT_EQ(act, exp);
}

TEST(Algorithms_Math_Operations__Test, cannot_multiply_matrix_and_vector_with_invalid_sizes) {
    Matrix<int> mat = {{1, 3, 5}, {2, 4, 6}};
    Vector<int> vec = {-1, 1};
    ASSERT_DEATH(mat * vec, ".*");
}

TEST(Algorithms_Math_Operations__Test, can_multiply_vector_and_matrix_with_valid_sizes) {
    Matrix<int> mat = {{1, 2}, {3, 4}, {5, 6}};
    Vector<int> vec = {-1, 0, 1};
    Vector<int> act = vec * mat;
    Vector<int> exp = {4, 4};
    ASSERT_EQ(act, exp);
}

TEST(Algorithms_Math_Operations__Test, can_multiply_vector_and_matrix_with_equal_sizes) {
    Matrix<int> mat = {{1, 2, -1}, {3, 4, 0}, {5, 6, 1}};
    Vector<int> vec = {13, 17, 15};
    Vector<int> act = vec * mat;
    Vector<int> exp = {139, 184, 2};
    ASSERT_EQ(act, exp);
}

TEST(Algorithms_Math_Operations__Test, cannot_multiply_vector_and_matrix_with_invalid_sizes) {
    Matrix<int> mat = {{1, 2}, {3, 4}, {5, 6}};
    Vector<int> vec = {-1, 1};
    ASSERT_DEATH(vec * mat, ".*");
}

TEST(Algorithms_Math_Operations__Test, can_invert_matrix) {
    Matrix<int> mat = {{1, 3, 5}, {-2, -4, -6}};
    Matrix<int> act = -mat;
    Matrix<int> exp = {{-1, -3, -5}, {2, 4, 6}};
    ASSERT_EQ(act, exp);
}

TEST(Algorithms_Math_Operations__Test, can_invert_vector) {
    Vector<int> vec = {1, -3, 5};
    Vector<int> act = -vec;
    Vector<int> exp = {-1, 3, -5};
    ASSERT_EQ(act, exp);
}

TEST(Algorithms_Math_Operations__Test, can_multiply_square_matrices_with_equal_sizes) {
    Matrix<int> mat1 = {{1, 2, -1}, {3, 4, 0}, {5, 6, 1}};
    Matrix<int> mat2 = {{7, 8, 9}, {-5, 5, 15}, {0, 1, 2}};
    Matrix<int> act = mat1 * mat2;
    Matrix<int> exp = {{-3, 17, 37}, {1, 44, 87}, {5, 71, 137}};
    ASSERT_EQ(act, exp);
}

TEST(Algorithms_Math_Operations__Test, cannot_multiply_square_matrices_with_different_sizes) {
    Matrix<int> mat1 = {{1, 2, -1}, {3, 4, 0}, {5, 6, 1}};
    Matrix<int> mat2 = {{7, 8}, {-5, 5}};
    ASSERT_DEATH(mat1 * mat2, ".*");
}

TEST(Algorithms_Math_Operations__Test, can_multiply_matrices_with_valid_different_sizes) {
    Matrix<int> mat1 = {{1, 2, -1}, {3, 4, 0}};
    Matrix<int> mat2 = {{1, 2}, {3, 4}, {5, 6}};
    Matrix<int> act = mat1 * mat2;
    Matrix<int> exp = {{2, 4}, {15, 22}};
    ASSERT_EQ(act, exp);
}

TEST(Algorithms_Math_Operations__Test, cannot_multiply_matrices_with_invalid_different_sizes) {
    Matrix<int> mat1 = {{1, 2, -1}, {3, 4, 0}};
    Matrix<int> mat2 = {{1, 2, -1}, {3, 4, 0}};
    ASSERT_DEATH(mat1 * mat2, ".*");
}
