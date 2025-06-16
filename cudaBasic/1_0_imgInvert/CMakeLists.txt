cmake_minimum_required(VERSION 3.18)
project(ImgInvert LANGUAGES CXX CUDA)

# Adding multiple architectures for portability
set(CMAKE_CUDA_ARCHITECTURES all-major)
message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable FIRST
add_executable(ImgInvert imgInvert.cu)

# THEN configure the target
target_include_directories(ImgInvert PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../utilities)
target_link_libraries(ImgInvert PRIVATE cuda cudart)

# Suppress STB library warnings for CUDA compiler
target_compile_options(ImgInvert PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress 550>
)

# Enable useful CUDA properties
set_target_properties(ImgInvert PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Print CUDA info when configuring
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")