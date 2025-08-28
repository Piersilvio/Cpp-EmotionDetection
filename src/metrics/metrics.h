#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

// Intersection over Union
float IoU(const cv::Rect& a, const cv::Rect& b);

// Precision e recall
void compute_metrics(const std::vector<cv::Rect>& predicted,
                     const std::vector<cv::Rect>& gt,
                     float iou_thresh,
                     float& precision,
                     float& recall,
                     int& tp,
                     int& fp,
                     int& fn);
