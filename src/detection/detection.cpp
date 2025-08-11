#include <opencv2/opencv.hpp>

#include "detection.h"
#include "../image/Image.h"

using namespace cv;

//Face classifier output
std::vector<Rect> detected_faces;

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


Image draw_face_box(Mat& input_image) {

    Image image_and_ROI;

    // For each face detected, draw a bounding box
    if (detected_faces.size() > 0) { 
        for (int i=0; i < detected_faces.size(); i++) {
            Rect r = detected_faces[i];

             // A color for each box
            cv::Scalar color = colors[i % colors.size()];

            cv::rectangle(input_image, Point(r.x, r.y), Point(r.x + r.width, r.y + r.height), color, 5, LINE_AA);

            Rect roi_coord(r.x, r.y,r.width,r.height);

            Mat roi_image = input_image(roi_coord);

            image_and_ROI.set_ROI(roi_image);
            image_and_ROI.set_pic(input_image);
        }

    }   

    return image_and_ROI;

}


void detect_face(Mat& input_image) {

    Mat gray_img;
    cvtColor( input_image, gray_img, COLOR_BGR2GRAY );
    equalizeHist( gray_img, gray_img );
    
    // Face classifier
    CascadeClassifier cascade;
    
    // Load the cascade classifier
    cascade.load(FACE_DETECTOR_MODEL_PATH);

    // Apply Viola-Jones in order to detect faces
    cascade.detectMultiScale(gray_img, detected_faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(100, 100));


}


std::vector<Rect> get_detected_faces(){
    return detected_faces;
}
