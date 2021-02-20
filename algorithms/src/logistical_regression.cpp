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
        double arg = y * (m_theta_0 + dot(m_theta, x.first));
        sum += std::log(sigmoid(arg));
    }
    // TODO: Here should be some extra calculations...
    return sum;
}

/*std::vector<double> LogisticalRegression::gradient(size_t i) const
{
    std::vector<double> grad(m_feature_count);
    const std::vector<double>& params = m_dataset[i].first;
    double arg = m_theta_0;
    for (size_t j = 0; j < m_feature_count; j++)
        arg += m_theta[j] * params[j];
    for (size_t j = 0; j < m_feature_count; j++)
        grad[j] = -(params[j] * std::exp(arg)) * sigmoid(arg);
    return grad;
}*/

std::vector<double> LogisticalRegression::gradient(size_t i, const std::vector<double>& x) const
{
    assert(x.size() == m_feature_count);
    std::vector<double> grad(m_feature_count);
    double arg = m_theta_0;
    for (size_t j = 0; j < m_feature_count; j++)
        arg += m_theta[j] * x[j];
    for (size_t j = 0; j < m_feature_count; j++)
        grad[j] = -(x[j] * std::exp(arg)) * sigmoid(arg);
    return grad;
}

double LogisticalRegression::minimal(size_t i) const
{
    std::vector<std::vector<double>> identity(m_feature_count);
    // Create identity matrix
    for (size_t i = 0; i < m_feature_count; i++)
    {
        identity[i].resize(m_feature_count, 0);
        identity[i][i] = 1;
    }
    std::vector<std::vector<double>> h = identity;
    double norm = 0;
    double epsilon = 1e-3;
    std::vector<double> x = m_dataset[i].first;
    do
    {
        std::vector<double> grad = gradient(i, x);
        std::vector<double> p(m_feature_count, 0);
        // p = -hessian x grad:
        for (size_t i = 0; i < m_feature_count; i++)
            for (size_t j = 0; j < m_feature_count; j++)
                p[i] -= h[i][j] * grad[j];

        double alpha = 0;
        // TODO: Search alpha to satisfy to Wolfe condition

        std::vector<double> x_next(m_feature_count);
        std::vector<double> x_delta(m_feature_count);
        for (size_t i = 0; i < m_feature_count; i++)
        {
            x_next[i] = x[i] + alpha * p[i];
            x_delta[i] = x_next[i] - x[i];
        }
        std::vector<double> grad_next = gradient(i, x_next);
        std::vector<double> grad_delta(m_feature_count);
        for (size_t i = 0; i < m_feature_count; i++)
            grad_delta[i] = grad_next[i] - grad[i];
        double ro = 1 / dot(grad_delta, x_delta);
        std::vector<std::vector<double>> h_next(m_feature_count);

        std::vector<std::vector<double>> temp = identity;
        for (size_t i = 0; i < m_feature_count; i++)
            for (size_t j = 0; j < m_feature_count; j++)
                temp[i][j] += grad_delta[j] * h[i][j];
        for (size_t i = 0; i < m_feature_count; i++)
            for (size_t j = 0; j < m_feature_count; j++)
                temp[i][j] += x_delta[j] * temp[i][j];
        for (size_t i = 0; i < m_feature_count; i++)
            for (size_t j = 0; j < m_feature_count; j++)
                temp[i][j] = ro * temp[i][j];
        for (size_t i = 0; i < m_feature_count; i++)
            for (size_t j = 0; j < m_feature_count; j++)
                temp[i][j] = h[i][j] - temp[i][j];
        
        std::vector<std::vector<double>> temp2 = temp;
        std::vector<double> vec(m_feature_count, 0);
        for (size_t i = 0; i < m_feature_count; i++)
            for (size_t j = 0; j < m_feature_count; j++)
                vec[i] += temp2[i][j] * grad_delta[j];
        for (size_t i = 0; i < m_feature_count; i++)
            for (size_t j = 0; j < m_feature_count; j++)
                temp[i][j] = ro * vec[i] * x_delta[j];
        for (size_t i = 0; i < m_feature_count; i++)
            for (size_t j = 0; j < m_feature_count; j++)
                temp[i][j] = temp2[i][j] - temp[i][j];
        
        std::vector<std::vector<double>> temp3 = identity;
        for (size_t i = 0; i < m_feature_count; i++)
        {
            for (size_t j = 0; j < m_feature_count; j++)
            {
                temp3[i][j] = ro * x_delta[i] * x_delta[j];
                h_next[i][j] = temp[i][j] + temp3[i][j];
            }
        }

        x = x_next;
        grad = grad_next;
        h = h_next;

        double sum = 0;
        for (size_t i = 0; i < m_feature_count; i++)
            sum += grad[i] * grad[i];
        norm = std::sqrt(sum);
    } while(norm > epsilon);
    // TODO: Define return value
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

    double arg = dot(m_theta, inputs) + m_theta_0;
    return sigmoid(arg);
}
