cmake_minimum_required(VERSION 3.31)
project(Add100s LANGUAGES CXX CUDA)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES all-major)

add_executable(add100Basic add100Basic.cu)
add_executable(add100Blocks add100Blocks.cu)

target_link_libraries(add100Basic PRIVATE cuda cudart)
set_target_properties(add100Basic PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(add100Blocks PRIVATE cuda cudart)
set_target_properties(add100Blocks PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Test takes a long time to run, so it's commented out
add_test(NAME test_add100Basic
         COMMAND add100Basic
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_test(NAME test_add100Blocks
         COMMAND add100Blocks
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})