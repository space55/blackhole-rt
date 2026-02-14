cmake_minimum_required(VERSION 3.18)

# --- Optional CUDA support ------------------------------------------------
# Enable with: cmake -DFLARESIM_USE_CUDA=ON ..
option(FLARESIM_USE_CUDA "Enable CUDA GPU acceleration for ghost rendering" OFF)

if(FLARESIM_USE_CUDA)
    project(flaresim LANGUAGES CXX CUDA)
else()
    project(flaresim LANGUAGES CXX)
endif()

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
if(ipo_supported AND NOT FLARESIM_USE_CUDA)
    # LTO can cause issues with mixed CXX/CUDA; enable only for CPU builds
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

if(FLARESIM_USE_CUDA)
    list(APPEND FLARESIM_SOURCES src/gpu_ghosts.cu)
endif()

# --- Flare simulator ------------------------------------------------------
add_executable(flaresim ${FLARESIM_SOURCES})
target_include_directories(flaresim PRIVATE src "${CMAKE_SOURCE_DIR}/../lib")
target_link_libraries(flaresim OpenEXR::OpenEXR)
if(OpenMP_CXX_FOUND)
    target_link_libraries(flaresim OpenMP::OpenMP_CXX)
endif()

if(FLARESIM_USE_CUDA)
    target_compile_definitions(flaresim PRIVATE FLARESIM_USE_CUDA)
    set_target_properties(flaresim PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "60;70;75;80;86;89;90"
    )
    # Ensure CUDA files see our headers
    target_include_directories(flaresim PRIVATE src)
    message(STATUS "CUDA ghost rendering: ENABLED")
else()
    message(STATUS "CUDA ghost rendering: DISABLED (use -DFLARESIM_USE_CUDA=ON to enable)")
endif()
