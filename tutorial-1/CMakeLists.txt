cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_BUILD_TYPE "")

project(yolov5 VERSION 0.1.0)

find_package(OpenCV 4.10.0 REQUIRED)

add_executable(yolov5 src/try-yolov5.cpp)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolov5 ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})
