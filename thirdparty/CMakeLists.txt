cmake_minimum_required(VERSION 3.14)

add_subdirectory(rapidcsv)

if(ENABLE_TESTS)
    add_subdirectory(googletest)
endif()
