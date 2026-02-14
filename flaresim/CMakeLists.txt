cmake_minimum_required(VERSION 3.18)
project(flaresim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

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

configure_file("flaresim.conf" "flaresim.conf" COPYONLY)

# --- Dependencies ---------------------------------------------------------
find_package(OpenEXR 3 REQUIRED)
find_package(OpenMP)

# --- Source files ---------------------------------------------------------
set(FLARESIM_SOURCES
    src/main.cpp
    src/lens.cpp
    src/trace.cpp
    src/ghost.cpp
    src/stb_impl.cpp
)

# --- Flare simulator ------------------------------------------------------
add_executable(flaresim ${FLARESIM_SOURCES})
target_include_directories(flaresim PRIVATE src "${CMAKE_SOURCE_DIR}/../lib")
target_link_libraries(flaresim OpenEXR::OpenEXR)
if(OpenMP_CXX_FOUND)
    target_link_libraries(flaresim OpenMP::OpenMP_CXX)
endif()
