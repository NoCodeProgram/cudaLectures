cmake_minimum_required(VERSION 3.14)
project(HelloWorld)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(HelloWorld main.cpp)

add_test(NAME test_helloWorld
         COMMAND HelloWorld
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})