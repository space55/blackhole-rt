cmake_minimum_required(VERSION 3.18)
project(flaretest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenEXR 3 REQUIRED)

add_executable(flaretest flaretest.cpp)
target_link_libraries(flaretest OpenEXR::OpenEXR)
