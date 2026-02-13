cmake_minimum_required(VERSION 3.18)
project(bhrt-tools LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Match main project optimisation flags
if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wextra")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")
endif()

# LTO
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# --- Dependencies ---------------------------------------------------------
find_package(OpenEXR 3 REQUIRED)
find_package(OpenMP)

# --- Bloom tool -----------------------------------------------------------
add_executable(bhrt-bloom bloom.cpp)
target_include_directories(bhrt-bloom PRIVATE "${CMAKE_SOURCE_DIR}/../lib")
target_link_libraries(bhrt-bloom OpenEXR::OpenEXR)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bhrt-bloom OpenMP::OpenMP_CXX)
endif()
