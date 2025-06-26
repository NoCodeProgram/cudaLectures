cmake_minimum_required(VERSION 3.31)
project(VerticalFlip LANGUAGES CXX CUDA)

# Set standards
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA configuration
set(CMAKE_CUDA_ARCHITECTURES all-major)

# Create executable
add_executable(verticalFlip verticalFlip.cu)

# Configure target
target_include_directories(verticalFlip PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../utilities)
target_link_libraries(verticalFlip PRIVATE cuda cudart)
target_compile_options(verticalFlip PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress 550>)

# CUDA properties
set_target_properties(verticalFlip PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../resources/cat32gray.png 
               ${CMAKE_CURRENT_BINARY_DIR}/cat32gray.png COPYONLY)

add_test(NAME test_verticalFlip 
         COMMAND verticalFlip
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})