/*
 *  Pre-processing of images containing ROIs, mapping of class ID to the corresponding emotion
 *  label, loading of the model, performing inference.
 */
 
#ifndef EMOTION_H
#define EMOTION_H

#include <opencv2/opencv.hpp>
#include <iostream>

#include "../image/Image.h"

// Preprocess ROIs before giving them in input to the model for emotion recognition
void preprocessROI(std::vector<cv::Mat>& _roi_image, Image& img);

// Load model and run model inference
std::vector<std::string> predict(Image& img, std::string model);

// Print the predicted emotion label
Image print_predicted_label(Image& image_and_ROI, std::vector<std::string>& emotion_prediction, std::vector<cv::Rect> detected_faces);

#endif

