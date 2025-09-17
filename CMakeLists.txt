cmake_minimum_required(VERSION 3.22)
project(FaceDetectionEmotionRecognition LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(ENV{OpenCV_DIR} "C:\\tools\\opencv\\build")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

if (MSVC)
    add_compile_options(/Zc:__cplusplus)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui dnn objdetect)
message(STATUS "Found OpenCV ${OpenCV_VERSION}")


add_executable(FaceDetectionEmotionRecognition
        src/Main.cpp
        src/detection/detection.cpp
        src/image/Image.cpp
        src/emotion_recognition/emotion_recognition.cpp
        src/emotion_recognition/pre-processing.cpp
        src/metrics/metrics.cpp
        src/utils/filesystem.cpp
        src/utils/selection.cpp
        src/utils/draw.cpp
        src/utils/pipeline.cpp
        src/utils/evaluation.cpp
)

target_include_directories(FaceDetectionEmotionRecognition PRIVATE
        src
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(FaceDetectionEmotionRecognition PRIVATE ${OpenCV_LIBS})
