#include <opencv2/opencv.hpp>

#include "detection.h"
#include "../image/Image.h"

using namespace cv;

//Face classifier output
std::vector<Rect> detected_faces;


Image draw_face_box(Mat& input_image) {

    Image image_and_ROI;

    // For each face detected, draw a bounding box
    if (detected_faces.size() > 0) { 
        for (int i=0; i < detected_faces.size(); i++) {
            Rect r = detected_faces[i];

            // draw rectangle around face
            rectangle(input_image,
                        Point(r.x, r.y),
                        Point(r.x + r.width, r.y + r.height),
                        Scalar(255, 0, 0), 3, 8, 0);

            Rect roi_coord(r.x, r.y,r.width,r.height);

            Mat roi_image = input_image(roi_coord);

            image_and_ROI.setROI(roi_image);
            image_and_ROI.setPic(input_image);
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
