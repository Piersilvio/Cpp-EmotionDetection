//Spicoli Piersilvio

#ifndef DRAW_H
#define DRAW_H



#include <opencv2/opencv.hpp>
#include <vector>

// Draws the ground-truth bounding boxes (red rectangles) on an image.
void draw_ground_truth(cv::Mat& img, const std::vector<cv::Rect>& boxes);



#endif //DRAW_H
