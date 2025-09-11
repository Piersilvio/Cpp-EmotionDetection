#include <opencv2/opencv.hpp>

#include "../image/Image.h"
#include "emotion_recognition.h"

using namespace cv;

void preprocessROI(std::vector<Mat>& roi_image, Image& image_and_ROI) {

    Mat processed_image;
    std::vector<Mat>  preprocessed_ROI;

    if (roi_image.empty()) {
        std::cerr << "No ROI found, skip preprocessing." << std::endl;
        image_and_ROI.set_preprocessed_ROI(preprocessed_ROI);
        return;
    }

    if (roi_image.size() > 0) { 
        for (int i=0; i < roi_image.size(); i++) {
        
                if (roi_image[i].empty()) {
            std::cerr << "ROI " << i << " empty, skipped." << std::endl;
            continue;
        }


                // === Preprocessing ===
                Mat gray;
                if (roi_image[i].channels() == 3) {
                    cvtColor(roi_image[i], gray, COLOR_BGR2GRAY);
                 } else {
                gray = roi_image[i];
                 }       

                 Mat resized;
                 resize(gray, resized, Size(48, 48));

                 Mat normalized;
                 resized.convertTo(normalized, CV_32F, 1.0 / 255.0);

                 roi_image[i] = normalized;
                // =====================
               
               
                // Append onto the model input vector
                preprocessed_ROI.push_back(roi_image[i]);
        }
    }
    
    image_and_ROI.set_preprocessed_ROI(preprocessed_ROI); 
    
    return;

}


