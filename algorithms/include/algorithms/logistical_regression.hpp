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
    double m_theta_0;

    static double sigmoid(double arg);
    static double dot(const std::vector<double>& x, const std::vector<double>& y);
    std::vector<double> gradient(size_t i, const std::vector<double>& x) const;

    double cost(size_t i) const;

    /**
     * Finds minimal value of ... with BFGS algorithm
     * 
     * @param i Index of dataset entry
     */
    double minimal(size_t i) const;
    
    void train();

public:
    LogisticalRegression(const std::vector<Pair>& dataset);
    ~LogisticalRegression() = default;

    double classify(const std::vector<double>& inputs);
};

}
