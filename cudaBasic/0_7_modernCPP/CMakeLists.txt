cmake_minimum_required(VERSION 3.31)
project(AddVecsModern LANGUAGES CXX CUDA)

# Adding multiple architectures for portability
set(CMAKE_CUDA_ARCHITECTURES all-major)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(addVecsModern addVecsModern.cu)
target_link_libraries(addVecsModern PRIVATE cuda cudart)
set_target_properties(addVecsModern PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_test(NAME test_addVecsModern 
         COMMAND addVecsModern
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})