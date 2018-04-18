cmake_minimum_required(VERSION 3.5)
set(PROJ_NAME homework_5)
file(GLOB SOURCES *.cpp *.hpp *.h *.cu)

project(${PROJ_NAME})

find_package(CUDA)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_35)
set(CMAKE_CXX_STANDARD 11)

include_directories(/usr/local/cuda-9.1/targets/x86_64-linux/include)
include_directories(lib)

cuda_add_executable(${PROJ_NAME} ${SOURCES})

