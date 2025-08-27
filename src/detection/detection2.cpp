#include <opencv2/opencv.hpp>
#include "detection.h"
#include "../image/Image.h"

using namespace cv;

// Face classifier output
std::vector<Rect> detected_faces;

// Colors
std::vector<cv::Scalar> colors = {
    cv::Scalar(255, 0, 0),
    cv::Scalar(0, 255, 0),
    cv::Scalar(0, 0, 255),
    cv::Scalar(255, 255, 0),
    cv::Scalar(255, 0, 255),
    cv::Scalar(0, 255, 255),
    cv::Scalar(128, 0, 128)
};

// IoU per rimuovere duplicati
float IoU(const Rect& a, const Rect& b) {
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);
    int interArea = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int unionArea = a.area() + b.area() - interArea;
    return unionArea > 0 ? (float)interArea / unionArea : 0.0f;
}

Image draw_face_box(Mat& input_image) {
    Image image_and_ROI;
    for (size_t i = 0; i < detected_faces.size(); i++) {
        Rect r = detected_faces[i];
        Scalar color = colors[i % colors.size()];
        rectangle(input_image, r, color, 5, LINE_AA);

        Mat roi_image = input_image(r);
        image_and_ROI.set_ROI(roi_image);
        image_and_ROI.set_pic(input_image);
    }
    return image_and_ROI;
}

void detect_face(Mat& input_image) {
    Mat gray_img;
    cvtColor(input_image, gray_img, COLOR_BGR2GRAY);
    equalizeHist(gray_img, gray_img);

    // Carica frontal + profile cascade
    CascadeClassifier frontal, profile;
    if (!frontal.load("../models/haarcascade_frontalface_alt2.xml")) {
        std::cerr << "Errore: frontal cascade non caricato\n";
        return;
    }
    if (!profile.load("../models/haarcascade_profileface.xml")) {
        std::cerr << "Errore: profile cascade non caricato\n";
        return;
    }

    std::vector<Rect> faces_frontal, faces_profile, all_faces;

    frontal.detectMultiScale(gray_img, faces_frontal, 1.1, 5, 0, Size(50,50));
    profile.detectMultiScale(gray_img, faces_profile, 1.1, 5, 0, Size(50,50));

    // Unisci risultati
    all_faces.insert(all_faces.end(), faces_frontal.begin(), faces_frontal.end());
    all_faces.insert(all_faces.end(), faces_profile.begin(), faces_profile.end());

    // Rimuovi duplicati usando IoU
    detected_faces.clear();
    for (const auto& f : all_faces) {
        bool keep = true;
        for (const auto& d : detected_faces) {
            if (IoU(f, d) > 0.5f) {
                keep = false;
                break;
            }
        }
        if (keep) detected_faces.push_back(f);
    }
}

std::vector<Rect> get_detected_faces() {
    return detected_faces;
}
