cmake_minimum_required(VERSION 3.31)
project(Block100Real LANGUAGES CXX CUDA)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES all-major)

add_executable(block100real block100real.cu)

target_link_libraries(block100real PRIVATE cuda cudart)
set_target_properties(block100real PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_test(NAME test_add100Basic_small
         COMMAND sh -c "echo '1000' | $<TARGET_FILE:block100real>"
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_test(NAME test_add100Basic_large
         COMMAND sh -c "echo '1000000' | $<TARGET_FILE:block100real>"
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})