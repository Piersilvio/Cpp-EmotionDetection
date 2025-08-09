#include <opencv2/opencv.hpp>

#include "../image/Image.h"
#include "emotion_recognition.h"

using namespace cv;

void preprocessROI(std::vector<Mat>& roi_image, Image& image_and_ROI) {

    Mat processed_image;
    std::vector<cv::Mat>  preprocessed_ROI;

        // Controllo se ci sono ROI
    if (roi_image.empty()) {
        std::cerr << "No ROI found, skip preprocessing." << std::endl;
        image_and_ROI.setPreprocessedROI(preprocessed_ROI);
        return;
    }

    if (roi_image.size() > 0) { 
        for (int i=0; i < roi_image.size(); i++) {
        
                if (roi_image[i].empty()) {
            std::cerr << "ROI " << i << " empty, skipped." << std::endl;
            continue;
        }

                // convert to grayscale 
                Mat gray_image;
                cvtColor( roi_image[i], gray_image, COLOR_BGR2GRAY );

                // Resize the ROI to model input size
                resize(gray_image, processed_image, Size(48,48));

                // Convert image pixels from between 0-255 to 0-1
                processed_image.convertTo(processed_image, CV_32FC3, 1.f/255);

                // Append onto the model input vector
                preprocessed_ROI.push_back(processed_image);
        }
    }
    
    image_and_ROI.setPreprocessedROI(preprocessed_ROI); 
    
    return;

}


