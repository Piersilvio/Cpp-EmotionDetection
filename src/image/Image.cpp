#include <opencv2/opencv.hpp>
#include <utility>

#include "image.h"

using namespace cv;

// Returns the stored image (_pic).
Mat Image::get_pic() {
    return this->_pic;
}

// Sets the stored image (_pic) to the given Mat.
void Image::set_pic(Mat& pic) {
    this->_pic = pic;
}

// Returns the vector of ROI images (_roi_image).
std::vector<Mat> Image::get_ROI() {
    return this->_roi_image;
}

// Adds a new ROI image to the vector _roi_image.
void Image::set_ROI(Mat& roi) {
    this->_roi_image.push_back(roi);
}

// Returns the vector of preprocessed ROI images (preprocessed_ROI).
std::vector<Mat> Image::get_preprocessed_ROI() {
    return this->preprocessed_ROI;
}

// Replaces the vector of preprocessed ROI images with a new one.
void Image::set_preprocessed_ROI(std::vector<Mat> prepr_roi) {
    this->preprocessed_ROI = std::move(prepr_roi);
}
