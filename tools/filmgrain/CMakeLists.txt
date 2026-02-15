cmake_minimum_required(VERSION 3.18)
project(filmgrain LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(filmgrain filmgrain.cpp)

# stb headers live in the main project's lib/ directory
target_include_directories(filmgrain PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../lib")
