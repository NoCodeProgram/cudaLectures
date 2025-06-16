cmake_minimum_required(VERSION 3.31)
project(CudaBasics LANGUAGES CXX CUDA)

enable_testing()

add_subdirectory(0_3_cppPrerequisites)
add_subdirectory(0_4_helloCUDA)
add_subdirectory(0_5_cudaSyntax)
add_subdirectory(0_6_addVecs)
add_subdirectory(0_7_modernCPP)
# add_subdirectory(0_8_cudaCheckErr)
# add_subdirectory(0_9_cudaDeviceCheck)
add_subdirectory(1_0_imgInvert)