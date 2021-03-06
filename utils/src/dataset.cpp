#include "dataset.hpp"

#include <rapidcsv.h>

using namespace Utils;

Dataset::Dataset(const std::string& filename, int label_row, int label_column)
    : doc(filename, rapidcsv::LabelParams(label_row, label_column)) {
}
