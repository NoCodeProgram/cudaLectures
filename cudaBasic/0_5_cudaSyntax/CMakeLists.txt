cmake_minimum_required(VERSION 3.18)
project(CUDA_Multiply LANGUAGES CXX CUDA)

# Adding multiple architectures for portability
set(CMAKE_CUDA_ARCHITECTURES all-major)
message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(cudaAdd100 cudaAdd100.cu)

# Link CUDA libraries
target_link_libraries(cudaAdd100 PRIVATE cuda cudart)

# Enable useful CUDA properties
set_target_properties(cudaAdd100 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Print CUDA info when configuring
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")