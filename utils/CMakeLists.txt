cmake_minimum_required(VERSION 3.14)

set(TARGET_NAME utils)

file(GLOB_RECURSE TARGET_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp
)

file(GLOB_RECURSE TARGET_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_library(${TARGET_NAME} STATIC ${TARGET_SRC} ${TARGET_HEADERS})

target_include_directories(${TARGET_NAME} PUBLIC
    include/utils
    ${CMAKE_SOURCE_DIR}/thirdparty/rapidcsv/src
)

target_link_libraries(${TARGET_NAME}
PUBLIC
    rapidcsv
)
