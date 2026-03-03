cmake_minimum_required(VERSION 3.18)
project(exrgrade LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenEXR 3 REQUIRED)

add_executable(exrgrade exrgrade.cpp)
target_link_libraries(exrgrade OpenEXR::OpenEXR)

# stb headers live in the main project's lib/ directory
target_include_directories(exrgrade PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../lib")
