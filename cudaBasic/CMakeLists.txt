cmake_minimum_required(VERSION 3.18)
project(CudaBasics LANGUAGES CXX CUDA)

# 서브디렉토리 추가
add_subdirectory(0_3_cppPrerequisites)
add_subdirectory(0_4_helloCUDA)
add_subdirectory(0_5_cudaSyntax)
add_subdirectory(0_6_addVecs)
add_subdirectory(0_7_modernCPP)
add_subdirectory(0_8_cudaCheckErr)
add_subdirectory(0_9_cudaDeviceCheck)
add_subdirectory(1_0_imgInvert)