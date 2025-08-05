#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

#include <opencv2/opencv.hpp>

#include "Image.h"

extern const std::string FACE_DETECTOR_MODEL_PATH;

/**
 * FaceDetector class does the face detection using cascade face detection classifier
 * Also this class prints the bounding box and prediction text to the image frame
 */
class FaceDetector {

public:
    // Constructor loads the cascade face detection classifier
    FaceDetector();

    // Function that runs face detection and draws the bounding box
    void detectFace(cv::Mat& frame);

    Image drawBoundingBoxOnFrame(cv::Mat& frame);

    // Function that print the predicted label to the image frame
    Image printPredictionTextToFrame( Image& image_and_ROI, std::vector<std::string>& emotion_prediction);

private:
    // Face detection classifier
    cv::CascadeClassifier cascade;

    // Output of the face detection classifier
    std::vector<cv::Rect> faces;

    //Colors
    std::vector<cv::Scalar> colors = {
    cv::Scalar(255, 0, 0),     // Blue
    cv::Scalar(0, 255, 0),     // Green
    cv::Scalar(0, 0, 255),     // Red
    cv::Scalar(255, 255, 0),   // Cyan
    cv::Scalar(255, 0, 255),   // Magenta
    cv::Scalar(0, 255, 255),   // Yellow
    cv::Scalar(128, 0, 128)    // Purple
};

};

#endif
