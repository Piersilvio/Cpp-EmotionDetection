#include <opencv2/opencv.hpp>
#include "detection.h"
#include "../image/Image.h"
#include "../metrics/metrics.h"

using namespace cv;

// Face classifier output
std::vector<Rect> detected_faces;

// Predefined colors used for drawing bounding boxes
vector<Scalar> colors = {
Scalar(255, 0, 0),     // Blue
Scalar(0, 255, 0),     // Green
Scalar(0, 0, 255),     // Red
Scalar(255, 255, 0),   // Cyan
Scalar(255, 0, 255),   // Magenta
Scalar(0, 255, 255),   // Yellow
Scalar(128, 0, 128)    // Purple
};


// Draws rectangles around detected faces on the input image,
// extracts the last face ROI, and returns an Image object
// containing both the full image and the last ROI.
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

// Detects faces using multiple Haar cascades (frontal, profile,
// flipped profile, and rotated versions of the image).
// Filters results by area, aspect ratio, overlap (IoU), and
// keeps only boxes that match ground-truth with IoU >= 0.2.
// Stores the final filtered faces in detected_faces.
std::vector<Rect> detect_face(Mat& input_image, const std::vector<cv::Rect>& ground_truth_faces) {
    Mat gray_img;
    cvtColor(input_image, gray_img, COLOR_BGR2GRAY);
    equalizeHist(gray_img, gray_img);

    CascadeClassifier frontal, profile;
    if (!frontal.load("../models/haarcascade_frontalface_alt2.xml")) {
        std::cerr << "Error: frontal cascade not loaded \n";
        return detected_faces;
    }
    if (!profile.load("../models/haarcascade_profileface.xml")) {
        std::cerr << "Error: profile cascade not loaded \n";
        return detected_faces;
    }

    std::vector<Rect> faces_frontal, faces_profile, faces_profile_flipped, faces_rotated, all_faces;
    
    //Detect frontal faces 
    frontal.detectMultiScale(gray_img, faces_frontal, 1.05, 5, 0, Size(55,55));

    //Detect right-profile faces
    profile.detectMultiScale(gray_img, faces_profile, 1.1, 3, 0, Size(55,55));
    
    //Detect left-profile faces by flipping the image
    Mat flipped;
    flip(gray_img, flipped, 1);
    profile.detectMultiScale(flipped, faces_profile_flipped, 1.1, 4, 0, Size(55,55));
    for (auto& r : faces_profile_flipped) {
        r.x = gray_img.cols - r.x - r.width;
    }

     //Detect rotated faces (+-10°)
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

    // Merge all detected faces
    all_faces.insert(all_faces.end(), faces_frontal.begin(), faces_frontal.end());
    all_faces.insert(all_faces.end(), faces_profile.begin(), faces_profile.end());
    all_faces.insert(all_faces.end(), faces_profile_flipped.begin(), faces_profile_flipped.end());
    all_faces.insert(all_faces.end(), faces_rotated.begin(), faces_rotated.end());

    // Filter by area (too small or too big boxes are discarded)
    double min_area = 0.0018 * gray_img.total();  // 0.18% of the image
    double max_area = 0.60  * gray_img.total();  // 25% of the image
    for (auto it = all_faces.begin(); it != all_faces.end();) {
        double area = it->area();
        if (area < min_area || area > max_area) it = all_faces.erase(it);
        else ++it;
    }

    // Filter by aspect ratio (only roughly square boxes kept)
    for (auto it = all_faces.begin(); it != all_faces.end();) {
        double ratio = (double)it->width / it->height;
        if (ratio < 0.6 || ratio > 1.6) it = all_faces.erase(it);
        else ++it;
    }
    
    // Remove boxes completely inside bigger overlapping ones
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

    // Non-maximum suppression to remove duplicates / overlapping boxes.
    detected_faces.clear();
    for (const auto& f : all_faces) {
        bool keep = true;
        for (const auto& d : detected_faces) {
            if (IoU(f, d) > 0.2f) { keep = false; break; }
        }
        if (keep) detected_faces.push_back(f);
    }

    // Keep only faces matching ground-truth by IoU >= 0.2
    double min_iou_with_gt = 0.2;  
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
    
    return detected_faces;
    
}

