cmake_minimum_required(VERSION 3.31)
project(HelloCuda LANGUAGES CXX CUDA)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES all-major)

add_executable(cudaCode cudaCode.cu)
add_executable(cpuCode cpuCode.cpp)

target_link_libraries(cudaCode PRIVATE cuda cudart)
set_target_properties(cudaCode PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Test takes a long time to run, so it's commented out
# add_test(NAME test_helloCUDA 
#          COMMAND cudaCode
#          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
# add_test(NAME test_helloCPU
#          COMMAND cpuCode
#          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})