cmake_minimum_required(VERSION 3.31)
project(Block2DBlur LANGUAGES CXX CUDA)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES all-major)

add_executable(block2DBlur block2DBlur.cu)

target_include_directories(block2DBlur PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../utilities)
target_link_libraries(block2DBlur PRIVATE cuda cudart)
target_compile_options(block2DBlur PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress 550>)

set_target_properties(block2DBlur PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../resources/cat1000gray.png
               ${CMAKE_CURRENT_BINARY_DIR}/cat1000gray.png COPYONLY)

add_test(NAME test_block2DBlur
         COMMAND block2DBlur
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
