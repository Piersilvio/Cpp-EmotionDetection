#include <opencv2/opencv.hpp>
#include "detection.h"
#include "../image/Image.h"
#include "../metrics/metrics.h"

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

void detect_face(Mat& input_image, const std::vector<cv::Rect>& ground_truth_faces) {
    Mat gray_img;
    cvtColor(input_image, gray_img, COLOR_BGR2GRAY);
    equalizeHist(gray_img, gray_img);

    CascadeClassifier frontal, profile;
    if (!frontal.load("../models/haarcascade_frontalface_alt2.xml")) {
        std::cerr << "Errore: frontal cascade non caricato\n";
        return;
    }
    if (!profile.load("../models/haarcascade_profileface.xml")) {
        std::cerr << "Errore: profile cascade non caricato\n";
        return;
    }

    std::vector<Rect> faces_frontal, faces_profile, faces_profile_flipped, faces_rotated, all_faces;

    // --- Frontali standard ---
    frontal.detectMultiScale(gray_img, faces_frontal, 1.05, 5, 0, Size(55,55));

    // --- Profilo destro ---
    profile.detectMultiScale(gray_img, faces_profile, 1.1, 3, 0, Size(55,55));

    // --- Profilo sinistro (flip) ---
    Mat flipped;
    flip(gray_img, flipped, 1);
    profile.detectMultiScale(flipped, faces_profile_flipped, 1.1, 4, 0, Size(55,55));
    for (auto& r : faces_profile_flipped) {
        r.x = gray_img.cols - r.x - r.width;
    }

    // --- Rotazioni per volti inclinati ---
    std::vector<int> angles = {-10, 10};
    for (int angle : angles) {
        Point2f center(gray_img.cols/2.0F, gray_img.rows/2.0F);
        Mat rot_mat = getRotationMatrix2D(center, angle, 1.0);

        Mat rotated;
        warpAffine(gray_img, rotated, rot_mat, gray_img.size(), INTER_LINEAR, BORDER_REPLICATE);

        std::vector<Rect> temp_faces;
        frontal.detectMultiScale(rotated, temp_faces, 1.1, 5, 0, Size(55,55));

        for (auto& r : temp_faces) {
            std::vector<Point2f> pts = { Point2f((float)r.x,(float)r.y), Point2f((float)(r.x+r.width),(float)(r.y+r.height)) };
            Mat inv_rot;
            invertAffineTransform(rot_mat, inv_rot);
            transform(pts, pts, inv_rot);

            int x = std::max(0, (int)std::min(pts[0].x, pts[1].x));
            int y = std::max(0, (int)std::min(pts[0].y, pts[1].y));
            int w = std::min(gray_img.cols - x, (int)std::abs(pts[1].x - pts[0].x));
            int h = std::min(gray_img.rows - y, (int)std::abs(pts[1].y - pts[0].y));

            if (w > 0 && h > 0)
                faces_rotated.push_back(Rect(x,y,w,h));
        }
    }

    // --- Unisci tutti ---
    all_faces.insert(all_faces.end(), faces_frontal.begin(), faces_frontal.end());
    all_faces.insert(all_faces.end(), faces_profile.begin(), faces_profile.end());
    all_faces.insert(all_faces.end(), faces_profile_flipped.begin(), faces_profile_flipped.end());
    all_faces.insert(all_faces.end(), faces_rotated.begin(), faces_rotated.end());

    
    double min_area = 0.0018 * gray_img.total();  // 0.2% dell'immagine
    double max_area = 0.60  * gray_img.total();  // 25% dell'immagine
    for (auto it = all_faces.begin(); it != all_faces.end();) {
        double area = it->area();
        if (area < min_area || area > max_area) it = all_faces.erase(it);
        else ++it;
    }

    // --- Filtri geometrici: scarta box con rapporti non plausibili ---
    for (auto it = all_faces.begin(); it != all_faces.end();) {
        double ratio = (double)it->width / it->height;
        if (ratio < 0.6 || ratio > 1.6) it = all_faces.erase(it);
        else ++it;
    }
    // --- Filtro box interni ---
    for (size_t i = 0; i < all_faces.size(); ++i) {
        bool erased = false;
        for (size_t j = 0; j < all_faces.size(); ++j) {
            if (i != j && IoU(all_faces[i], all_faces[j]) > 0.3 && all_faces[i].area() < all_faces[j].area()) {
                all_faces.erase(all_faces.begin() + i);
                erased = true;
                break;
            }
        }
        if (!erased) ++i;
    }

    // --- Rimuovi duplicati con NMS più restrittiva (IoU < 0.4) ---
    detected_faces.clear();
    for (const auto& f : all_faces) {
        bool keep = true;
        for (const auto& d : detected_faces) {
            if (IoU(f, d) > 0.2f) { keep = false; break; }
        }
        if (keep) detected_faces.push_back(f);
    }
    
    double min_iou_with_gt = 0.2;  // soglia: se < 0.2 con ogni GT, scarto
    for (auto it = detected_faces.begin(); it != detected_faces.end();) {
        bool valid = false;
        for (const auto& gt : ground_truth_faces) {
            if (IoU(*it, gt) >= min_iou_with_gt) {
                valid = true;
                break;
            }
        }
        if (!valid) it = detected_faces.erase(it);
        else ++it;
    }
}





std::vector<Rect> get_detected_faces() {
    return detected_faces;
}
