#include <opencv2/opencv.hpp>

#include "detection.h"
#include "../image/Image.h"

using namespace cv;

//Face classifier output
std::vector<Rect> detected_faces;


Image draw_face_box(Mat& frame) {

    Image image_and_ROI;

    // For each face detected, draw a bounding box
    if (detected_faces.size() > 0) { 
        for (int i=0; i < detected_faces.size(); i++) {
            Rect r = detected_faces[i];

            // draw rectangle around face
            rectangle(frame,
                        Point(r.x, r.y),
                        Point(r.x + r.width, r.y + r.height),
                        Scalar(255, 0, 0), 3, 8, 0);

            Rect roi_coord(r.x, r.y,r.width,r.height);

            Mat roi_image = frame(roi_coord);

            image_and_ROI.setROI(roi_image);
            image_and_ROI.setPic(frame);
        }

    }   

    return image_and_ROI;

}

Image print_predicted_label( Image& image_and_ROI, std::vector<std::string>& emotion_prediction) {

    Mat img = image_and_ROI.getPic();
    
    if (detected_faces.size() > 0) { 
        for (int i=0; i < detected_faces.size(); i++) {
            Rect r = detected_faces[i];

            // Write text prediction on bounding box
            putText(img, //target image
                        emotion_prediction[i], //text - will take the output of the model.inference()
                        Point(r.x, r.y-10), //top-left position of box
                        FONT_HERSHEY_DUPLEX,
                        1.0,
                        CV_RGB(118, 185, 0), //font color
                        2);
        }
    }

    image_and_ROI.setPic(img);

    return image_and_ROI;

}

void detect_face(Mat& frame) {

    Mat gray_img;
    cvtColor( frame, gray_img, COLOR_BGR2GRAY );
    equalizeHist( gray_img, gray_img );
    
    // Face classifier
    CascadeClassifier cascade;
    
    // Load the cascade classifier
    cascade.load(FACE_DETECTOR_MODEL_PATH);

    // Apply Viola-Jones in order to detect faces
    cascade.detectMultiScale(gray_img, detected_faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(100, 100));


}
