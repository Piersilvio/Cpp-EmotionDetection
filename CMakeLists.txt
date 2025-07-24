cmake_minimum_required(VERSION 3.22)
project(EmotionDetectionProject)

set(CMAKE_CXX_STANDARD 23)

set(ENV{OPENCV_DIR} "C:\\tools\\opencv\\build")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(EmotionDetectionProject
        src/main.cpp
        src/FaceDetector.cpp
        src/Image.cpp
        src/Model.cpp)



target_link_libraries(EmotionDetectionProject ${OpenCV_LIBS})
