cmake_minimum_required(VERSION 3.22)
project(EmotionDetectionProject LANGUAGES CXX)

# C++ standard (17 è sufficiente per <filesystem>; puoi lasciare 23 se vuoi)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# (Opzionale ma utile su MSVC per avere __cplusplus corretto)
if (MSVC)
    add_compile_options(/Zc:__cplusplus)
endif()

# Se serve, indica esplicitamente dove sta OpenCVConfig.cmake:
# set(OpenCV_DIR "C:/tools/opencv/build")

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui dnn objdetect)
message(STATUS "Found OpenCV ${OpenCV_VERSION}")

add_executable(EmotionDetectionProject
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

# Per #include "utils/..." ecc.
target_include_directories(EmotionDetectionProject PRIVATE
        src
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(EmotionDetectionProject PRIVATE ${OpenCV_LIBS})
