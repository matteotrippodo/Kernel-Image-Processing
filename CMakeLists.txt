cmake_minimum_required(VERSION 3.8)
project(Kernel_Image_Processing CUDA)

set(CMAKE_CXX_STANDARD 17)

add_executable(Kernel_Image_Processing main.cu)

set_target_properties(Kernel_Image_Processing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(Kernel_Image_Processing PROPERTIES CUDA_ARCHITECTURES "50;72")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(Kernel_Image_Processing ${OpenCV_LIBS})







