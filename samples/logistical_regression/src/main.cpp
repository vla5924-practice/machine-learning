#include <iostream>
#include <string>

#include <algorithms/logistical_regression.hpp>
#include <utils/dataset.hpp>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: ./logistical_regression_sample PATH_TO_TRAINING_DATASET [PATH_TO_TESTING_DATASET]"
                  << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    Utils::Dataset dataset(filename);
    auto data = dataset.asPairsWithVector<double, bool>();

    Algorithms::LogisticalRegression log_reg(data);

    if (argc == 3) {
        std::string inputs_filename = argv[2];
        Utils::Dataset inputs_data(inputs_filename);
        auto inputs = inputs_data.asVectors<double>();
        for (const auto& input : inputs) {
            for (double feature : input)
                std::cout << feature << ',';
            std::cout << log_reg.classify(input) << std::endl;
        }
        return 0;
    }

    while (true) {
        std::cout << "Enter input features (" << log_reg.featureCount() << "): ";
        std::vector<double> inputs(log_reg.featureCount());
        for (size_t i = 0; i < log_reg.featureCount(); i++)
            std::cin >> inputs[i];
        std::cout << "Classification result (OR): " << log_reg.classify(inputs) << std::endl;
    }
}
