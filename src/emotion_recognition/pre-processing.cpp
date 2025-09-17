// Author: Jacopo Onorati

/*
 *  Pre-processing of the images containing ROIs
 */
 
#include <opencv2/opencv.hpp>

#include "../image/Image.h"
#include "emotion_recognition.h"

using namespace cv;

void preprocessROI(std::vector<Mat>& ROI_images, Image& img) {

    Mat processed_image;
    std::vector<Mat>  preprocessed_ROI;

    if (ROI_images.empty()) {
        std::cerr << "No ROI found, skip preprocessing." << std::endl;
        img.set_preprocessed_ROI(preprocessed_ROI);
        return;
    }

    if (ROI_images.size() > 0) { 
        for (int i=0; i < ROI_images.size(); i++) {
        
            if (ROI_images[i].empty()) {
                std::cerr << "ROI " << i << " empty, skipped." << std::endl;
                continue;
            }

            // Convert to grayscale 
            Mat gray_image;
            cvtColor(ROI_images[i], gray_image, COLOR_BGR2GRAY );

            // Resize the ROI to model input size
            resize(gray_image, processed_image, Size(48,48));

            // Convert image pixels from between 0-255 to 0-1
            processed_image.convertTo(processed_image, CV_32FC3, 1.f/255);
            
            preprocessed_ROI.push_back(processed_image);
        }
    }
    
    img.set_preprocessed_ROI(preprocessed_ROI); 
    
    return;

}


