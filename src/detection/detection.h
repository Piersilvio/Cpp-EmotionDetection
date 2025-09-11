#ifndef FACEDETECTION_H
#define FACEDETECTION_H

#include <opencv2/opencv.hpp>

#include "../image/Image.h"

extern const std::string FACE_DETECTOR_MODEL_PATH;


// Detect faces
extern void detect_face(cv::Mat& input_image,const std::vector<cv::Rect>& ground_truth_faces);
    
// Draw a box sorrounding the detected face
Image draw_face_box(cv::Mat& input_image);

std::vector<cv::Rect> get_detected_faces(void);

#endif
