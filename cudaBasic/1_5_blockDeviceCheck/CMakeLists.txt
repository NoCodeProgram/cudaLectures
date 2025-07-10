cmake_minimum_required(VERSION 3.31)
project(BlockDeviceCheck LANGUAGES CXX CUDA)

# Adding multiple architectures for portability
set(CMAKE_CUDA_ARCHITECTURES all-major)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(blockDeviceCheck blockDeviceCheck.cu)
target_link_libraries(blockDeviceCheck PRIVATE cuda cudart)
set_target_properties(blockDeviceCheck PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_test(NAME test_blockDeviceCheck
         COMMAND blockDeviceCheck
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
