/*
 * Face detection using Viola-Jones algorithm. Post-processing: application of a sequence filters: 
 * removal of bounding boxes with unrealistic sizes, exclusion of boxes with implausible aspect
 * ratios, suppression of nested detections where a smaller box lies within a larger one and 
 * non-maximum suppression to eliminate duplicate detections.
 */

#ifndef FACEDETECTION_H
#define FACEDETECTION_H

#include <opencv2/opencv.hpp>

#include "../image/Image.h"

// Detect faces
extern std::vector<cv::Rect> detect_face(cv::Mat& input_image,const std::vector<cv::Rect>& ground_truth_faces);
    
// Draw a box sorrounding the detected face
Image draw_face_box(cv::Mat& input_image);

#endif
