cmake_minimum_required(VERSION 3.14)

set(TARGET_NAME logistical_regression_sample)

file(GLOB_RECURSE TARGET_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_executable(${TARGET_NAME} ${TARGET_SRC})

target_include_directories(${TARGET_NAME} PUBLIC
    ${CMAKE_SOURCE_DIR}/algorithms/include
    ${CMAKE_SOURCE_DIR}/utils/include
)

target_link_libraries(${TARGET_NAME}
PUBLIC
    algorithms
    utils
)
