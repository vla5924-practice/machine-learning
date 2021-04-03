#include "logistical_regression.hpp"

#include "maths/functions.hpp"
#include "maths/operations.hpp"
#include "maths/types.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

using namespace Algorithms;
using namespace Algorithms::Maths;

double LogisticalRegression::sigmoid(double arg) {
    return 1. / (1. + std::exp(-arg));
}

double LogisticalRegression::convertTarget(bool target) {
    return target ? 1 : -1;
}

double LogisticalRegression::cost(const VectorExtended& theta) const {
    double sum = 0;
    for (const Pair& data : m_dataset) 
      sum += std::log(1 + std::exp(-convertTarget(data.second) * (theta.second + dot(theta.first, data.first))));
    sum *= m_params.c;
    sum += 0.5 * (1 - m_params.ro) * (theta.second * theta.second + dot(theta.first, theta.first));
    sum += m_params.ro * norm(theta.first);
    return sum;
}

std::vector<double> LogisticalRegression::gradient() const {
    std::vector<double> grad(m_feature_count);
    for (size_t j = 0; j < m_feature_count; j++) {
        grad[j] = 0;
        for (size_t i = 0; i < m_dataset.size(); i++) {
            double target = convertTarget(m_dataset[i].second);
            grad[j] += m_params.c / (1 + exp(-target * (m_theta.second + dot(m_theta.first, m_dataset[i].first)))) *
                       exp(-target * (m_theta.second + dot(m_theta.first, m_dataset[i].first))) *
                       (-target * m_dataset[i].first[j]);
        }
        double sign = m_theta.first[j] > 0 ? 1 : (m_theta.first[j] < 0 ? -1 : 0);
        grad[j] += 0.5 * (1 - m_params.ro) * 2 * m_theta.first[j] + m_params.ro * sign;
    }
    double grad_0 = 0;
    for (size_t i = 0; i < m_dataset.size(); i++) {
        double target = convertTarget(m_dataset[i].second);
        grad_0 += m_params.c / (1 + exp(-target * (m_theta.second + dot(m_theta.first, m_dataset[i].first)))) *
                  exp(-target * (m_theta.second + dot(m_theta.first, m_dataset[i].first))) * (-target);
    }
    double sign = m_theta.second > 0 ? 1 : (m_theta.second < 0 ? -1 : 0);
    grad_0 += 0.5 * (1 - m_params.ro) * 2 * m_theta.second + m_params.ro * sign;
    grad.insert(grad.begin(), grad_0);
    return grad;
}

std::vector<double> LogisticalRegression::minimal(const std::vector<double>& x_start) {
    const size_t size_basic = m_feature_count;
    const size_t size_extended = size_basic + 1u;
    assert(x_start.size() == size_extended);
    const Matrix<double> identity = createIdentity<double>(size_extended);
    Matrix<double> h = identity;
    Vector<double> x_curr = x_start;
    Vector<double> grad = gradient();
    do {
        Vector<double> p = -h * grad;
        double alpha = 0.1;
        // TODO: Search alpha to satisfy to Wolfe condition
        double x_0 = x_curr[0];
        Vector<double> x_temp = {std::next(x_curr.begin()), x_curr.end()};
        Vector<double> x_next = x_curr + alpha * p;
        m_theta.second = x_next[0];
        std::copy(std::next(x_next.begin()), x_next.end(), m_theta.first.begin());
        double cost_l = cost(m_theta);
        x_next = x_curr - alpha * p;
        x_0 = x_curr[0];
        x_temp = {std::next(x_next.begin()), x_next.end()};
        double cost_r = cost({x_temp, x_0});
        if (cost_l <= cost_r) {
            x_next = x_next = x_curr + alpha * p;
        } else {
            m_theta.second = x_next[0];
            std::copy(std::next(x_curr.begin()), x_curr.end(), m_theta.first.begin());
        }
        Vector<double> x_delta = x_next - x_curr;
        Vector<double> grad_next = gradient();
        Vector<double> grad_delta = grad_next - grad;
        double ro = 1 / dot(grad_delta, x_delta);
        Matrix<double> h_next = h + (x_delta * (-ro * (grad_delta * h)));
        h_next = h_next + ((-ro * (h * grad_delta)) * x_delta);
        double ro2 = ro * ro * dot(grad_delta * h, grad_delta);
        h_next = h_next + (ro2 * (x_delta * x_delta));
        h_next = h_next + (ro * (x_delta * x_delta));
        assert(ndims(h) == ndims(h_next));
        x_curr = x_next;
        grad = grad_next;
        h = h_next;
    } while (norm(grad) > m_params.epsilon);
    return x_curr;
}

void LogisticalRegression::train() {
    m_theta.second = 0;
    m_theta.first.resize(m_feature_count, 0);
    minimal(Vector<double>(m_feature_count + 1u, 0.0));
}

LogisticalRegression::LogisticalRegression(const std::vector<Pair>& dataset, const Params& params) : m_params(params) {
    if (dataset.empty())
        throw std::runtime_error("Empty dataset is not allowed");

    m_feature_count = dataset.front().first.size();
    for (const Pair& entry : dataset)
        if (entry.first.size() != m_feature_count)
            throw std::runtime_error("Dataset entries have different sizes");

    m_dataset = dataset;
    m_theta.second = 0;
    train();
}

size_t LogisticalRegression::featureCount() const {
    return m_feature_count;
}

std::pair<std::vector<double>, double> LogisticalRegression::dividingHyperplane() const {
    return m_theta;
}

double LogisticalRegression::classify(const std::vector<double>& inputs) {
    if (inputs.size() != m_feature_count)
        throw std::runtime_error("Input does not have the same size as dataset");

    double arg = m_theta.second + dot(m_theta.first, inputs);
    return sigmoid(arg);
}
