#include <vector>

namespace Algorithms
{

/**
 * Logistical regression classifier
 * 
 * @todo Description
 */
class LogisticalRegression
{
public:
    /**
     * Single dataset entry
     * 
     * std::vector<double> Vector of feature values
     * bool                Target value (does feature belong to class or not)
     */
    using Pair = std::pair<std::vector<double>, bool>;

protected:
    // Count of features in dataset
    size_t m_feature_count;

    // Dataset for model to be pre-trained
    std::vector<Pair> m_dataset;

    // Coefficients calculated with dataset given for pre-training
    std::vector<double> m_theta;

    static double sigmoid(double x);

    void train();

public:
    LogisticalRegression(const std::vector<Pair>& dataset);
    ~LogisticalRegression() = default;

    double classify(const std::vector<double>& inputs);
};

}
