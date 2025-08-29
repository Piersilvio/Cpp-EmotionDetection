#include "metrics.h"
#include "metrics.h"
#include <fstream>   
#include <sstream>
#include <iostream>  
#include <filesystem>
namespace fs = std::filesystem;

using namespace cv;

// Funzione per leggere bounding box YOLO-style
std::vector<cv::Rect> read_ground_truth(const std::string& label_file, int img_width, int img_height) {
    std::vector<cv::Rect> boxes;
    std::ifstream infile(label_file);  // ora compilatore sa cos’è ifstream
    if (!infile.is_open()) return boxes;

    std::string line;
    while (getline(infile, line)) {
        std::istringstream ss(line);
        int class_id;
        float x_center, y_center, w, h;
        ss >> class_id >> x_center >> y_center >> w >> h;

        int x1 = static_cast<int>((x_center - w/2.0f) * img_width);
        int y1 = static_cast<int>((y_center - h/2.0f) * img_height);
        int width = static_cast<int>(w * img_width);
        int height = static_cast<int>(h * img_height);

        boxes.push_back(cv::Rect(x1, y1, width, height));
    }
    return boxes;
}


// IoU
float IoU(const Rect& a, const Rect& b) {
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);

    int interArea = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int unionArea = a.area() + b.area() - interArea;
    return unionArea > 0 ? (float)interArea / unionArea : 0.0f;
}

// Precision e recall
void compute_metrics(const std::vector<cv::Rect>& predicted,
                     const std::vector<cv::Rect>& gt,
                     float iou_thresh,
                     float& precision,
                     float& recall,
                     int& tp,
                     int& fp,
                     int& fn) {

    tp = 0; fp = 0; fn = 0;
    std::vector<bool> gt_matched(gt.size(), false);

    for (const auto& p : predicted) {
        bool matched = false;
        for (size_t i = 0; i < gt.size(); i++) {
            if (!gt_matched[i] && IoU(p, gt[i]) >= iou_thresh) {
                tp++;
                gt_matched[i] = true;
                matched = true;
                break;
            }
        }
        if (!matched) fp++;
    }

    for (bool m : gt_matched)
        if (!m) fn++;

    precision = tp + fp > 0 ? (float)tp / (tp + fp) : 0.0f;
    recall = tp + fn > 0 ? (float)tp / (tp + fn) : 0.0f;
}

string clean_pred_label(const string& raw_label) {
    size_t pos = raw_label.find(":");
    if (pos != string::npos) return raw_label.substr(0, pos);
    return raw_label;
}

string extract_gt_label(const string& filename) {
    string stem = fs::path(filename).stem().string();
    size_t pos = stem.find("(");
    if (pos != string::npos) stem = stem.substr(0, pos);
    return stem;
}

string normalize_label(const string& s) {
    string out = s;
    out.erase(remove_if(out.begin(), out.end(), ::isspace), out.end());
    transform(out.begin(), out.end(), out.begin(), ::tolower);
    return out;
}
