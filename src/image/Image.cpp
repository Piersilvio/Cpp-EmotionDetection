#include <opencv2/opencv.hpp>

#include "image.h"

using namespace cv;

Mat Image::getPic() {
    return this->_pic;
}

void Image::setPic(Mat& pic) {
    this->_pic = pic;
}

std::vector<Mat> Image::getROI() {
    return this->_roi_image;
}

void Image::setROI(Mat& roi) {
    this->_roi_image.push_back(roi);
}

std::vector<Mat> Image::getPreprocessedROI() {
    return this->preprocessed_ROI;
}

void Image::setPreprocessedROI(std::vector<Mat> prepr_roi) {
    this->preprocessed_ROI = prepr_roi;
}
