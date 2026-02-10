cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# GPU acceleration via CUDA (off by default)
option(USE_GPU "Build with CUDA GPU acceleration" OFF)

if(USE_GPU)
    project(bhrt3 LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    # Target common GPU architectures (Volta, Ampere, Ada Lovelace, Hopper)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "70;80;89")
    endif()
else()
    project(bhrt3 LANGUAGES CXX)
endif()

# Link-Time Optimization: enables cross-TU inlining of hot functions
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

find_package(OpenMP)

if (CMAKE_BUILD_TYPE MATCHES Debug)
    message("Building Debug build")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -pg -Wall -Wextra")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")
endif()

#configure_file("config.toml" "config.toml" COPYONLY)
configure_file("hubble-skymap.jpg" "hubble-skymap.jpg" COPYONLY)
configure_file("scene.txt" "scene.txt" COPYONLY)

set(SOURCES
        src/main.cpp
        src/scene.cpp
        src/sky.cpp
        src/stb_impl.cpp
)

if(USE_GPU)
    list(APPEND SOURCES gpu/gpu_render.cu)
endif()

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE "inc/")
target_include_directories(${PROJECT_NAME} PRIVATE "lib/")
target_include_directories(${PROJECT_NAME} PRIVATE "gpu/")

if(USE_GPU)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_GPU)
    # Fast math for CUDA device code; -lineinfo for profiling
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()