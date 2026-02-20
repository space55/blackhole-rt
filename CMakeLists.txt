cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# GPU acceleration via CUDA (off by default)
option(USE_GPU "Build with CUDA GPU acceleration" OFF)

# GPU acceleration via Metal (macOS / Apple Silicon)
option(USE_METAL "Build with Metal GPU acceleration (macOS)" OFF)

# Mutual exclusion: CUDA and Metal cannot both be active
if(USE_GPU AND USE_METAL)
    message(FATAL_ERROR "USE_GPU (CUDA) and USE_METAL cannot both be enabled")
endif()

# Single-precision mode: use float instead of double for all ray integration
# and physics.  ~2x faster on GPU (32x on consumer cards), slightly less
# accurate near the photon sphere.  Compare with the fp64 default.
option(USE_FLOAT "Use float (fp32) instead of double (fp64) for physics" OFF)

if(USE_GPU)
    project(bhrt3 LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    # Target common GPU architectures (Volta, Ampere, Ada Lovelace, Hopper)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "70;80;89")
    endif()
elseif(USE_METAL)
    project(bhrt3 LANGUAGES CXX OBJCXX)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 20)
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
find_package(OpenEXR 3 QUIET)
if(OpenEXR_FOUND)
    message(STATUS "Found OpenEXR: ${OpenEXR_VERSION}")
else()
    message(STATUS "OpenEXR not found — EXR output will be disabled")
endif()

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

if(USE_METAL)
    list(APPEND SOURCES gpu/metal_render.mm)
endif()

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE "inc/")
target_include_directories(${PROJECT_NAME} PRIVATE "lib/")
target_include_directories(${PROJECT_NAME} PRIVATE "gpu/")

if(USE_GPU)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_GPU)
    # Fast math for CUDA device code; -lineinfo for profiling;
    # --expt-relaxed-constexpr allows constexpr host functions (<cmath>) in
    # device code — safety net for any residual mixed-type math calls.
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>)
endif()

if(USE_METAL)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_METAL BH_USE_FLOAT)
    # Link Metal and Foundation frameworks
    target_link_libraries(${PROJECT_NAME} "-framework Metal" "-framework Foundation")
    # ObjC++ flags for the .mm file
    set_source_files_properties(gpu/metal_render.mm PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
    )
    # Compile .metal shader to .metallib at build time
    # Requires full Xcode (not just Command Line Tools) for the Metal compiler.
    find_program(METAL_COMPILER xcrun)
    set(METAL_SOURCE "${CMAKE_SOURCE_DIR}/gpu/metal_render.metal")
    set(METAL_AIR "${CMAKE_BINARY_DIR}/metal_render.air")
    set(METAL_LIB "${CMAKE_BINARY_DIR}/metal_render.metallib")
    if(METAL_COMPILER)
        # Test if the metal compiler is actually available
        execute_process(
            COMMAND xcrun --find metal
            RESULT_VARIABLE METAL_FOUND
            OUTPUT_QUIET ERROR_QUIET
        )
    endif()
    if(METAL_COMPILER AND METAL_FOUND EQUAL 0)
        add_custom_command(
            OUTPUT ${METAL_AIR}
            COMMAND xcrun -sdk macosx metal
                -c ${METAL_SOURCE}
                -o ${METAL_AIR}
                -I ${CMAKE_SOURCE_DIR}/inc
                -I ${CMAKE_SOURCE_DIR}
                -std=metal3.0
                -O2
                -DBH_USE_FLOAT
            DEPENDS ${METAL_SOURCE}
                ${CMAKE_SOURCE_DIR}/inc/physics.h
                ${CMAKE_SOURCE_DIR}/inc/vec_math.h
            COMMENT "Compiling Metal shader → .air"
        )
        add_custom_command(
            OUTPUT ${METAL_LIB}
            COMMAND xcrun -sdk macosx metallib
                ${METAL_AIR}
                -o ${METAL_LIB}
            DEPENDS ${METAL_AIR}
            COMMENT "Linking Metal shader → .metallib"
        )
        add_custom_target(metal_shaders ALL DEPENDS ${METAL_LIB})
        add_dependencies(${PROJECT_NAME} metal_shaders)
    else()
        message(WARNING "Metal compiler not found (install full Xcode). "
                        "The C++ host code will build but you must compile "
                        "metal_render.metal manually:\n"
                        "  xcrun -sdk macosx metal -c gpu/metal_render.metal "
                        "-o build/metal_render.air -I inc -std=metal3.0 -O2 -DBH_USE_FLOAT\n"
                        "  xcrun -sdk macosx metallib build/metal_render.air "
                        "-o build/metal_render.metallib")
    endif()
    message(STATUS "Metal GPU acceleration enabled (fp32 forced)")
endif()

if(USE_FLOAT)
    target_compile_definitions(${PROJECT_NAME} PRIVATE BH_USE_FLOAT)
    message(STATUS "Precision: FLOAT (fp32)")
else()
    message(STATUS "Precision: DOUBLE (fp64)")
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()

if(OpenEXR_FOUND)
    target_compile_definitions(${PROJECT_NAME} PRIVATE HAS_OPENEXR)
    target_link_libraries(${PROJECT_NAME} OpenEXR::OpenEXR)
endif()