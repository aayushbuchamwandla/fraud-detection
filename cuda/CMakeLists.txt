# Standalone CUDA build for the fused FraudMLP kernel -- independent of
# Python/PyTorch, so the kernel can be verified/benchmarked as a pure
# C++/CUDA artifact. This is also the library cpp/ (Phase 7) links against,
# so the inference math is defined exactly once.
#
# Build (inside WSL2, with CUDA Toolkit 11.3 + g++-10 -- see README):
#   cmake -S cuda -B cuda/build -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10
#   cmake --build cuda/build
#   ./cuda/build/kernel_correctness_test

cmake_minimum_required(VERSION 3.18)
project(fraud_cuda_kernel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3060 (Ampere, compute capability 8.6)

# Static library: the fused forward-pass kernel, reused by the correctness
# test below and by cpp/ (Phase 7).
add_library(fraud_kernel STATIC fraud_kernel.cu)
target_include_directories(fraud_kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(fraud_kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Correctness + benchmark test: loads models/exported/*.bin, runs a plain
# C++ reference forward pass and the CUDA kernel on the same input, and
# reports max absolute difference plus timing for both.
add_executable(kernel_correctness_test kernel_correctness_test.cu)
target_link_libraries(kernel_correctness_test PRIVATE fraud_kernel)
target_include_directories(kernel_correctness_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
