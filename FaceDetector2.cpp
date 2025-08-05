#include <opencv2/opencv.hpp>

#include "FaceDetector.h"
#include "Image.h"

FaceDetector::FaceDetector() {

     // Load the cascade classifier
    cascade.load(FACE_DETECTOR_MODEL_PATH);

}

void FaceDetector::detectFace(cv::Mat& frame) {

    cv::Mat gray_img;
    cv::cvtColor( frame, gray_img, cv::COLOR_BGR2GRAY );
    cv::equalizeHist( gray_img, gray_img );

    // Detect faces
    cascade.detectMultiScale(gray_img, this->faces, 1.1, 2, 0|cv::CASCADE_SCALE_IMAGE, cv::Size(100, 100));

}

Image FaceDetector::drawBoundingBoxOnFrame(cv::Mat& frame) {
    Image image_and_ROI;

    if (faces.size() > 0) {
        for (int i = 0; i < faces.size(); i++) {
            cv::Rect r = faces[i];

            // Colore diverso per ogni riquadro
            cv::Scalar color = colors[i % colors.size()];

            // Riquadro più spesso
            cv::rectangle(frame, cv::Point(r.x, r.y), cv::Point(r.x + r.width, r.y + r.height), color, 5, cv::LINE_AA);

            // Salva ROI
            cv::Rect roi_coord(r.x, r.y, r.width, r.height);
            cv::Mat roi_image = frame(roi_coord);

            image_and_ROI.setROI(roi_image);
            image_and_ROI.setFrame(frame);
        }
    }

    return image_and_ROI;
}
Image FaceDetector::printPredictionTextToFrame(Image& image_and_ROI, std::vector<std::string>& emotion_prediction) {
    cv::Mat img = image_and_ROI.getFrame();
    
    if (faces.size() > 0) { 
          for (int i = 0; i < faces.size(); i++) {
              cv::Rect r = faces[i];
          
              // Converti in maiuscolo
              std::string emotion_upper = emotion_prediction[i];
              std::transform(emotion_upper.begin(), emotion_upper.end(), emotion_upper.begin(), ::toupper);
          
              // Testo con font elegante e meno spesso
              cv::putText(img,
                          emotion_upper,
                          cv::Point(r.x, r.y - 10),
                          cv::FONT_HERSHEY_COMPLEX,
                          1.0,                          // scala
                          colors[i % colors.size()],    // stesso colore del riquadro
                          1);                           // spessore ridotto
          }
    }

    image_and_ROI.setFrame(img);
    return image_and_ROI;
}
