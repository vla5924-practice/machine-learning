#include "logistical_regression.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

using namespace Algorithms;

double LogisticalRegression::sigmoid(double arg)
{
    return 1. / (1. + std::exp(-arg));
}


double LogisticalRegression::dot(const std::vector<double>& x, const std::vector<double>& y)
{
    assert(x.size() == y.size());

    double sum = 0;
    for (size_t i = 0; i < x.size(); i++)
        sum += x[i] * y[i];
    return sum;
}

double LogisticalRegression::cost(size_t i) const
{
    double y = m_dataset[i].second ? -1 : 1;
    double sum = 0;
    for (const Pair& x : m_dataset)
    {
        double arg = y * (m_theta_free_member * dot(m_theta, x.first));
        sum += std::log(sigmoid(arg));
    }
    // TODO: Here should be some extra calculations...
    return sum;
}

double LogisticalRegression::gradient(size_t i) const
{
    return 0;
}

double LogisticalRegression::minimal(size_t i) const
{
    return 0;
}

void LogisticalRegression::train()
{

}

LogisticalRegression::LogisticalRegression(const std::vector<Pair>& dataset) {
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

    double arg = dot(m_theta, inputs) + m_theta_free_member;
    return sigmoid(arg);
}
