# C++ inference component -- independently buildable (does not require
# cuda/ to have been built first): compiles cuda/fraud_kernel.cu directly
# as part of this project, so the exact same kernel source is reused
# without duplicating the math, but this directory has no build-order
# dependency on cuda/CMakeLists.txt.
#
# Build (inside WSL2, with CUDA Toolkit 11.3 + g++-10 -- see README):
#   cmake -S cpp -B cpp/build -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10
#   cmake --build cpp/build
#   ./cpp/build/fraud_cpp_demo <weights_dir> <samples_csv>
#   ./cpp/build/fraud_cpp_demo <weights_dir> --benchmark <output_json>
#   ./cpp/build/test_predictor <weights_dir> <samples_csv>

cmake_minimum_required(VERSION 3.18)
project(fraud_cpp_inference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3060 (Ampere, compute capability 8.6)

set(CUDA_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../cuda)

# inference.cpp is compiled as plain C++ (not the CUDA language), so it
# needs cuda_runtime.h's include dir and the cudart library explicit --
# CMake only wires those up automatically for .cu sources.
find_package(CUDAToolkit REQUIRED)

# Inference library: FraudPredictor (C++) + the fused forward kernel (CUDA),
# compiled together as one target.
add_library(fraud_inference STATIC
    inference.cpp
    ${CUDA_SRC_DIR}/fraud_kernel.cu
)
target_include_directories(fraud_inference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_SRC_DIR}
)
set_target_properties(fraud_inference PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(fraud_inference PUBLIC CUDA::cudart)

# CLI demo + benchmark executable
add_executable(fraud_cpp_demo main.cpp)
target_link_libraries(fraud_cpp_demo PRIVATE fraud_inference)

# Correctness test executable
add_executable(test_predictor test_predictor.cpp)
target_link_libraries(test_predictor PRIVATE fraud_inference)
