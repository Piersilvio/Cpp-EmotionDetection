#ifndef IMAGE_H
#define IMAGE_H

#include <opencv2/opencv.hpp>
#include <iostream>




/**
 * An Image object contains the original picture and the detected regions of interest,
 * that can be setted and retrieved separately 
 */
class Image {

public: 
    //Constructor
    Image() {};
    //Destructor
    ~Image() {};
    
    
    cv::Mat getPic();
    void setPic(cv::Mat& pic);
    std::vector<cv::Mat> getROI();
    void setROI(cv::Mat& roi);
    std::vector<cv::Mat> getPreprocessedROI();
    void setPreprocessedROI(std::vector<cv::Mat> prepr_roi);
    
private:
    // the full video pic
    cv::Mat _pic;
    // region of interest within the bounding box
    std::vector<cv::Mat> _roi_image;
    // preprocessed image ready for model
    std::vector<cv::Mat>  preprocessed_ROI;
    

};

#endif
