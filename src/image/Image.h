#ifndef IMAGE_H
#define IMAGE_H

#include <opencv2/opencv.hpp>
#include <iostream>


/*
 * An Image object contains the original picture, the detected regions of interest and
 * the image with preprocessedn ROIs. These three elements can be setted and retrieved 
 * separately. 
 */
class Image {

public: 
    //Constructor
    Image() {};
    
    //Destructor
    ~Image() {};
    
    cv::Mat get_pic();
    void set_pic(cv::Mat& pic);
    std::vector<cv::Mat> get_ROI();
    void set_ROI(cv::Mat& roi);
    std::vector<cv::Mat> get_preprocessed_ROI();
    void set_preprocessed_ROI(std::vector<cv::Mat> prepr_roi);
    
private:
    
    // The image without ROIs
    cv::Mat _pic;   
    
    // Regions of interest within the bounding box
    std::vector<cv::Mat> _roi_image;
    
    // Preprocessed image ready for model
    std::vector<cv::Mat>  preprocessed_ROI;
    

};

#endif
