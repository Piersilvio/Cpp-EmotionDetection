#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <algorithm>
#include <mutex>

#include "detection/detection.h"
#include "emotion_recognition/emotion_recognition.h"

#include "image/Image.h"

using namespace std;
using namespace cv;


const string FACE_DETECTOR_MODEL_PATH = "../model/haarcascade_frontalface_alt2.xml";
const string TENSORFLOW_MODEL_PATH = "../model/tensorflow_model.pb";

const string window_name = "Face detection and emotion recognition";

const string image_path = "../images/surprise_5.jpg";
//const string image_path = "../images/happy_4.jpg";



int main()
{
    
    Mat image = imread(image_path);    

    Image image_and_ROI;

    namedWindow(window_name);

    // Detect faces applying Viola-Jones algorithm
    detect_face(image);

    // Draw a bounding box corresponding to the detected face 
    image_and_ROI = draw_face_box(image);

    // Get Image ROIs
    vector<Mat> roi_image = image_and_ROI.getROI();
    
    if (roi_image.size()>0) {
        // Preprocess image ready for model
        preprocessROI(roi_image, image_and_ROI);

        // Make Prediction
        vector<Rect> detected_faces = get_detected_faces();
        vector<string> emotion_prediction = predict(image_and_ROI, TENSORFLOW_MODEL_PATH);
        // Add prediction text to the output video image
        image_and_ROI = print_predicted_label(image_and_ROI, emotion_prediction, detected_faces);
    }

    Mat output_image = image_and_ROI.getPic();

    if (!output_image.empty()) {
        imshow (window_name, output_image);
    } 
    else {
       // if the output image is empty (ie. the facedetector didn't detect anything), just display the original image
       imshow (window_name, image);
    }
    waitKey(0);
     
    return 0;

}
