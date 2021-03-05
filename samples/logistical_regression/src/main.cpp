#include <iostream>
#include <string>

#include <algorithms/logistical_regression.hpp>
#include <utils/dataset.hpp>

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: ./logistical_regression_sample path_to_dataset.csv" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    Utils::Dataset dataset(filename);
    auto data = dataset.asPairsWithVector<double, bool>();

    Algorithms::LogisticalRegression log_reg(data);
    while (true)
    {
        std::cout << "Enter input features (" << log_reg.featureCount() << "): ";
        std::vector<double> inputs(log_reg.featureCount());
        for (size_t i = 0; i < log_reg.featureCount(); i++)
            std::cin >> inputs[i];
        std::cout << "Classification result (OR): " << log_reg.classify(inputs) << std::endl;
    }
}
