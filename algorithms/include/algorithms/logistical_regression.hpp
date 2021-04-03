#pragma once

#include <vector>

namespace Algorithms {

/**
 * Logistical regression classifier
 *
 * @todo Description
 */
class LogisticalRegression {
  public:
    /**
     * Single dataset entry
     *
     * std::vector<double> Vector of feature values
     * bool                Target value (does feature belong to class or not)
     */
    using Pair = std::pair<std::vector<double>, bool>;

    using VectorExtended = std::pair<std::vector<double>, double>;

    struct Params {
        double epsilon = 1e-5;
        double c = 5.0;
        double ro = 0;
        size_t iter_max = 10000u;
    };

  protected:
    // Count of features in dataset
    size_t m_feature_count;

    // Dataset for model to be pre-trained
    std::vector<Pair> m_dataset;

    // Coefficients calculated with dataset given for pre-training
    VectorExtended m_theta;

    Params m_params;

    static double sigmoid(double arg);
    static double convertTarget(bool target);

    double cost(const VectorExtended& theta) const;
    std::vector<double> gradient() const;
    std::vector<double> minimal(const std::vector<double>& x_start);

    void train();

  public:
    LogisticalRegression(const std::vector<Pair>& dataset, const Params& params = Params());
    ~LogisticalRegression() = default;

    size_t featureCount() const;

    VectorExtended dividingHyperplane() const;

    double classify(const std::vector<double>& inputs);
};

} // namespace Algorithms
