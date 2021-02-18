#include <vector>

namespace Algorithms
{

class LogisticalRegression
{
public:
    using Pair = std::pair<std::vector<double>, bool>;

protected:
    std::vector<Pair> m_training;

public:
    LogisticalRegression(const std::vector<Pair>& training);
    ~LogisticalRegression() = default;

    std::vector<double> classify(const std::vector<Pair>& inputs);
};

}
