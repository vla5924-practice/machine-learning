cmake_minimum_required(VERSION 3.14)

project(MachineLearning LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/archive)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(ENABLE_BENCHMARK "Enables benchmark app" OFF)
option(ENABLE_SAMPLES "Enables samples for algorithms" ON)

add_subdirectory(algorithms)
add_subdirectory(thirdparty)
add_subdirectory(utils)

if(ENABLE_BENCHMARK)
    add_subdirectory(benchmark)
endif()

if(ENABLE_SAMPLES)
    add_subdirectory(samples)
endif()
