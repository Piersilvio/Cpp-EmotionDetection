#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

#include <opencv2/opencv.hpp>

#include "../image/Image.h"

extern const std::string FACE_DETECTOR_MODEL_PATH;

// Detect faces
extern void detect_face(cv::Mat& frame);
    
// Draw a box sorrounding the detected face
Image draw_face_box(cv::Mat& frame);

// Print the predicted emotion label
Image print_predicted_label( Image& image_and_ROI, std::vector<std::string>& emotion_prediction);

#endif
