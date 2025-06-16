cmake_minimum_required(VERSION 3.18)
project(AddVecs LANGUAGES CXX CUDA)

# Adding multiple architectures for portability
set(CMAKE_CUDA_ARCHITECTURES all-major)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(addVecs addVecs.cu)
target_link_libraries(addVecs PRIVATE cuda cudart)
set_target_properties(addVecs PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_test(NAME test_addVecs 
         COMMAND addVecs
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})