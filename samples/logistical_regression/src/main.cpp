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
    Algorithms::LogisticalRegression algo(data);
}
