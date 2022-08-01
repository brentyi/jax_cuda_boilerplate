cmake_minimum_required(VERSION 3.18)
project(cuda_ray_jax LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)

find_package(pybind11 NO_MODULE REQUIRED)
find_package(Eigen3 3.3 NO_MODULE REQUIRED)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

include_directories(${CMAKE_CURRENT_LIST_DIR}/lib)

pybind11_add_module(
  ops
  ${CMAKE_CURRENT_LIST_DIR}/lib/kernels.cu
  ${CMAKE_CURRENT_LIST_DIR}/lib/bindings.cc)
install(TARGETS ops DESTINATION ${CMAKE_PROJECT_NAME})
set_target_properties(ops PROPERTIES CXX_STANDARD 17)
