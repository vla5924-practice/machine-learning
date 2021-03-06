cmake_minimum_required(VERSION 3.14)

project(MachineLearning LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/archive)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(ENABLE_BENCHMARK "Enables benchmark app" OFF)
option(ENABLE_SAMPLES "Enables samples for algorithms" ON)
option(ENABLE_TESTS "Enables tests for components" ON)

if(WIN32)
    set(CMAKE_CXX_FLAGS_DEBUG "/MTd /Z7 /Od")
    option(gtest_force_shared_crt "" TRUE)
endif()

add_subdirectory(algorithms)
add_subdirectory(thirdparty)
add_subdirectory(utils)

if(ENABLE_BENCHMARK)
    add_subdirectory(benchmark)
endif()

if(ENABLE_SAMPLES)
    add_subdirectory(samples)
endif()

if(ENABLE_TESTS)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(tests)
endif()
