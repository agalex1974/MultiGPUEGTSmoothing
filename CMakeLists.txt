cmake_minimum_required(VERSION 3.20)

set(default_build_type "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
set (CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

project(KNNCUDA CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CGAL_DISABLE_GMP TRUE)
find_package(CGAL REQUIRED)
include(${CGAL_USE_FILE})

include(CheckLanguage)
check_language(CUDA)
find_package(OpenMP)

add_executable(KNNCUDA main.cpp EGT_GKNN.cu EGT_GKNN.cuh File.cu GridKNN.cu GPU_Normal.cuh GPU_Normal.cu EigenSolver.cuh MST.h MST.cpp CUDA_MST.cu CUDA_MST.cuh parallel.h bfs.cu bfs.cuh
        miniply.cpp
        Mesh.h
        definitions.h
        MeshPLY.cpp
        MeshPLY.h
        GridKNNCuda.cu
        KNNInterface.cuh
        GridMeshCuda1.cu
        GridMeshCuda1.cuh
        EGT_CUDA.cu
        EGT_CUDA.cuh
        getOutputPoints.cu
        getOutputPoints.cuh
        Normal_CUDA.cu
        Normal_CUDA.cuh
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(KNNCUDA PUBLIC OpenMP::OpenMP_CXX)
    target_compile_options(KNNCUDA PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)
endif()
