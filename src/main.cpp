#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <algorithm>
#include <mutex>

#include "FaceDetector.h"
#include "Image.h"
#include "Model.h"

// Update this with the path to your opencv directory
const std::string FACE_DETECTOR_MODEL_PATH = "../model/haarcascade_frontalface_alt2.xml";
const std::string TENSORFLOW_MODEL_PATH = "../model/tensorflow_model.pb";
const std::string APP_NAME = "Facial Emotion Recognition";

const std::string IMAGE_PATH = "../images/surprise_5.jpg";

int main()
{
    // Initialise video frame that will be read from the camera
    cv::Mat frame = cv::imread(IMAGE_PATH);
    // Initialise all the required objects
    Model model(TENSORFLOW_MODEL_PATH);
    FaceDetector face_detector;
    Image image_and_ROI;

    //create a window with the window name
    cv::namedWindow(APP_NAME);


        //Run Face Detection and draw bounding box
        face_detector.detectFace(frame);

        // Draw bounding box to frame
        image_and_ROI = face_detector.drawBoundingBoxOnFrame(frame);

        // Get Image ROIs
        std::vector<cv::Mat> roi_image = image_and_ROI.getROI();
    
        if (roi_image.size()>0) {
            // Preprocess image ready for model
            image_and_ROI.preprocessROI();
            // Make Prediction
            std::vector<std::string> emotion_prediction = model.predict(image_and_ROI);
            // Add prediction text to the output video frame
            image_and_ROI = face_detector.printPredictionTextToFrame(image_and_ROI, emotion_prediction);
        }

        cv::Mat output_frame = image_and_ROI.getFrame();

        if (!output_frame.empty()) {
            imshow (APP_NAME, output_frame);
        } else {
            // if the output frame is empty (ie. the facedetector didn't detect anything), just display the original image
            imshow (APP_NAME, frame);
        }

        // wait for for 10 ms until any key is pressed.  
        // if the 'Esc' key is pressed, break the program loop
        cv::waitKey(0);
        //if (cv::waitKey(100) == 27)
        //{
          //  std::cout << "Esc key is pressed by user. Stopping the program" << std::endl;
        //}

    
    return 0;

}
