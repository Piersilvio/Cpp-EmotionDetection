#ifndef EMOTION_H
#define EMOTION_H



#include <opencv2/opencv.hpp>
#include <iostream>

#include "../image/Image.h"

/**
 * Model class that contains code to read the pretrained tensorflow model and allows
 * us to make predictions on new images
 */

    // Constructor reads in a pretrained (in python) tensorflow graph and weights (.pb file contains everything we need about the model)
    // Also initialises the mapping from class id to the string label (ie. happy, angry, sad etc)  



// Model inference function takes image input and outputs the prediction label and the probability
std::vector<std::string> predict(Image& image, std::string model_filename);


// Function to preprocess image for model input
void preprocessROI(std::vector<cv::Mat>& _roi_image, Image& image_and_ROI);

// Print the predicted emotion label
Image print_predicted_label(Image& image_and_ROI, std::vector<std::string>& emotion_prediction, std::vector<cv::Rect> detected_faces);


#endif

