cmake_minimum_required(VERSION 3.31)
project(Block2DGrayscale LANGUAGES CXX CUDA)

# Adding multiple architectures for portability
set(CMAKE_CUDA_ARCHITECTURES all-major)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(block2DGrayscale block2DGrayscale.cu)

target_include_directories(block2DGrayscale PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../utilities)
target_link_libraries(block2DGrayscale PRIVATE cuda cudart)
target_compile_options(block2DGrayscale PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress 550>)

set_target_properties(block2DGrayscale PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../resources/cat1024color.png 
               ${CMAKE_CURRENT_BINARY_DIR}/cat1024color.png COPYONLY)

add_test(NAME test_block2DGrayscale
         COMMAND block2DGrayscale
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
