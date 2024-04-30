cmake_minimum_required(VERSION 3.25.0)
project(train_gpt2cu LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")
set(CMAKE_CUDA_ARCHITECTURES "native")

include(cmake/CPM.cmake)
CPMAddPackage("gh:NVIDIA/cccl#main")
CPMAddPackage("gh:NVIDIA/nvbench#main")

find_package(OpenMP REQUIRED)

add_executable(train_gpt2 train_gpt2.cpp)
target_link_libraries(train_gpt2 PRIVATE OpenMP::OpenMP_CXX)

find_package(CUDAToolkit)
add_executable(train_gpt2cu train_gpt2.cu)
target_link_libraries(train_gpt2cu CCCL::CCCL CUDA::cublas CUDA::cublasLt)
target_compile_options(train_gpt2cu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --extended-lambda>)

add_subdirectory(benches)
