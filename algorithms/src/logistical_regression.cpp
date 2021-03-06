#include "logistical_regression.hpp"

#include "maths/functions.hpp"
#include "maths/operations.hpp"
#include "maths/types.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

using namespace Algorithms;
using namespace Algorithms::Maths;

double LogisticalRegression::sigmoid(double arg)
{
    return 1. / (1. + std::exp(-arg));
}

double LogisticalRegression::cost(size_t i) const
{
    double y = m_dataset[i].second ? -1 : 1;
    double sum = 0;
    for (const Pair& x : m_dataset)
    {
        double arg = y * (m_theta_0 + dot(m_theta, x.first));
        sum += std::log(sigmoid(arg));
    }
    // TODO: Here should be some extra calculations...
    return sum;
}

std::vector<double> LogisticalRegression::gradient(const std::vector<double>& x) const
{
    assert(x.size() == m_feature_count);
    std::vector<double> grad(m_feature_count);
    double arg = m_theta_0 + dot(m_theta, x);
    for (size_t i = 0; i < m_feature_count; i++)
        grad[i] = -(x[i] * std::exp(arg)) * sigmoid(arg);
    return grad;
}

std::vector<double> LogisticalRegression::minimal(size_t i) const
{
    Matrix<double> h = createIdentity<double>(m_feature_count);
    double epsilon = 1e-5;
    Vector<double> x = m_dataset[i].first;
    Vector<double> grad = gradient(x);
    do
    {
        Vector<double> p = -h * grad;
        double alpha = 1;
        // TODO: Search alpha to satisfy to Wolfe condition
        Vector<double> x_next = x + alpha * p;
        Vector<double> x_delta = x_next - x;
        Vector<double> grad_next = gradient(x_next);
        Vector<double> grad_delta = grad_next - grad;
        double ro = 1 / dot(grad_delta, x_delta);
        Matrix<double> temp = h - ro * (x_delta * (grad_delta * h));
        Matrix<double> h_next = (temp - ro * (temp * grad_delta) * x_delta) + (ro * x_delta * x_delta);
        x = x_next;
        grad = grad_next;
        h = h_next;
    } while(norm(grad) > epsilon);
    return x;
}

void LogisticalRegression::train()
{
    m_theta_0 = 0;
    m_theta.resize(m_feature_count, 0);
    std::vector<double> features(m_feature_count, 0);
    for (size_t i = 0; i < m_feature_count; i++)
        features = features + m_dataset[i].first;
    for (size_t i = 0; i < m_feature_count; i++)
        features[i] /= m_feature_count;
    std::vector<double> thetas(m_feature_count, 0);
    for (size_t i = 0; i < m_feature_count; i++)
        thetas = thetas + minimal(i);
    for (size_t i = 0; i < m_feature_count; i++)
        m_theta[i] = thetas[i] / m_feature_count;
}

LogisticalRegression::LogisticalRegression(const std::vector<Pair>& dataset) {
    if (dataset.empty())
        throw std::runtime_error("Empty dataset is not allowed");

    m_feature_count = dataset.front().first.size();
    for (const Pair& entry : dataset)
        if (entry.first.size() != m_feature_count)
            throw std::runtime_error("Dataset entries have different sizes");

    m_dataset = dataset;
    m_theta_0 = 0;
    train();
}

size_t LogisticalRegression::featureCount() const
{
    return m_feature_count;
}

std::pair<std::vector<double>, double> LogisticalRegression::dividingHyperplane() const
{
    return std::make_pair(m_theta, m_theta_0);
}

double LogisticalRegression::classify(const std::vector<double>& inputs)
{
    if (inputs.size() != m_feature_count)
        throw std::runtime_error("Input does not have the same size as dataset");

    double arg = m_theta_0 + dot(m_theta, inputs);
    return sigmoid(arg);
}
