cmake_minimum_required(VERSION 3.31)
project(GrayImg LANGUAGES CXX CUDA)

# Set standards
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA configuration
set(CMAKE_CUDA_ARCHITECTURES all-major)

# Create executable
add_executable(grayImg grayImg.cu)

# Configure target
target_include_directories(grayImg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../utilities)
target_link_libraries(grayImg PRIVATE cuda cudart)
target_compile_options(grayImg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress 550>)

# CUDA properties
set_target_properties(grayImg PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../resources/cat32color.png 
               ${CMAKE_CURRENT_BINARY_DIR}/cat32color.png COPYONLY)

add_test(NAME test_grayImg 
         COMMAND grayImg
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})