#include <opencv2/opencv.hpp>

#include "image.h"

using namespace cv;

Mat Image::get_pic() {
    return this->_pic;
}

void Image::set_pic(Mat& pic) {
    this->_pic = pic;
}

std::vector<Mat> Image::get_ROI() {
    return this->_roi_image;
}

void Image::set_ROI(Mat& roi) {
    this->_roi_image.push_back(roi);
}

std::vector<Mat> Image::get_preprocessed_ROI() {
    return this->preprocessed_ROI;
}

void Image::set_preprocessed_ROI(std::vector<Mat> prepr_roi) {
    this->preprocessed_ROI = prepr_roi;
}
