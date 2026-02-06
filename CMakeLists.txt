cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# set the project name
project(bhrt3 LANGUAGES CXX)

find_package(OpenMP)

if (CMAKE_BUILD_TYPE MATCHES Debug)
    message("Building Debug build")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -pg -Wall -Wextra")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
endif()

#configure_file("config.toml" "config.toml" COPYONLY)
configure_file("hubble-skymap.jpg" "hubble-skymap.jpg" COPYONLY)

set(SOURCES
        src/main.cpp
        src/sky.cpp
        src/ray.cpp
        src/stb_impl.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE "inc/")
target_include_directories(${PROJECT_NAME} PRIVATE "lib/")

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()