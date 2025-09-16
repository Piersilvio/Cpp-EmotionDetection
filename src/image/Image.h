#ifndef IMAGE_H
#define IMAGE_H

#include <opencv2/opencv.hpp>
#include <iostream>


/*
 * An Image object contains the final picture containing the results of  the application
 * of face detection and emotion recognition plus the ground truth boxes, the detected regions 
 * of interest and the images with preprocessed ROIs. These three elements can be setted and 
 * retrieved separately. 
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
    
    // Vector that contains images with Regions Of Interest (ROIs)  (an element for each detected ROI)
    std::vector<cv::Mat> _roi_image;
    
    // Vector that contains the images with ROIs preprocessed before to being fed into the model
    std::vector<cv::Mat>  preprocessed_ROI;
    

};

#endif
