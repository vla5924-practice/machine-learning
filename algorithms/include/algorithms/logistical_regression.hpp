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

    std::vector<double> gradient(const std::vector<double>& x) const;

    double cost(size_t i) const;

    /**
     * Finds minimal value of dataset-based cost function with BFGS algorithm
     * 
     * @param i Index of dataset entry
     */
    std::vector<double> minimal(size_t i) const;
    
    void train();

public:
    LogisticalRegression(const std::vector<Pair>& dataset);
    ~LogisticalRegression() = default;

    size_t featureCount() const;

    std::pair<std::vector<double>, double> dividingHyperplane() const;

    double classify(const std::vector<double>& inputs);
};

} // namespace Algorithms
