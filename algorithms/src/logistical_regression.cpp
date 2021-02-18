#include "logistical_regression.hpp"

#include <cmath>
#include <stdexcept>

using namespace Algorithms;

double LogisticalRegression::sigmoid(double x)
{
    return 1. / (1. + std::exp(-x));
}

void LogisticalRegression::train()
{

}

LogisticalRegression::LogisticalRegression(const std::vector<Pair>& dataset)
{
    if (dataset.empty())
        throw std::runtime_error("Empty dataset is not allowed");

    m_feature_count = dataset.front().first.size();
    for (const Pair& entry : dataset)
        if (entry.first.size() != m_feature_count)
            throw std::runtime_error("Dataset entries have different sizes");

    m_dataset = dataset;
    train();
}

double LogisticalRegression::classify(const std::vector<double>& inputs)
{
    if (inputs.size() != m_feature_count)
        throw std::runtime_error("Input does not have the same size as dataset");

    double prod = 0;
    for (size_t i = 0; i < m_feature_count; i++)
        prod += m_theta[i] * inputs[i];

    return sigmoid(prod);
}
