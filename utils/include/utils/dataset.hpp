#pragma once

#include <string>
#include <type_traits>
#include <vector>

#include <rapidcsv.h>

namespace Utils {

class Dataset {
    rapidcsv::Document doc;

  public:
    Dataset(const std::string& filename, int label_row = -1, int label_column = -1);
    ~Dataset() = default;

    template <typename T>
    std::vector<std::vector<T>> asVectors() {
        std::vector<std::vector<T>> data;
        size_t rows = doc.GetRowCount();
        for (size_t i = 0; i < rows; i++)
            data.push_back(doc.GetRow<T>(i));
        return data;
    }

    template <typename T, typename Tx>
    std::vector<std::pair<std::vector<T>, Tx>> asPairsWithVector() {
        size_t rows = doc.GetRowCount();
        size_t cols = doc.GetColumnCount();
        std::vector<std::pair<std::vector<T>, Tx>> data(0);
        data.reserve(rows);
        for (size_t i = 0; i < rows; i++) {
            std::vector<T> part(cols - 1);
            for (size_t j = 0; j < cols - 1; j++)
                part[j] = doc.GetCell<T>(j, i);
            if (std::is_same<Tx, bool>::value)
                data.emplace_back(part, doc.GetCell<int>(cols - 1, i));
            else
                data.emplace_back(part, doc.GetCell<Tx>(cols - 1, i));
        }
        return data;
    }
};

} // namespace Utils
