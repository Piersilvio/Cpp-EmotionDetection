#include "metrics.h"
using namespace cv;

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
