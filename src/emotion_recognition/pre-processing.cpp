#include <opencv2/opencv.hpp>
#include "../image/Image.h"
#include "emotion_recognition.h"

using namespace cv;

void preprocessROI(std::vector<Mat>& roi_image, Image& image_and_ROI) {

    std::vector<Mat> preprocessed_ROI;

    if (roi_image.empty()) {
        std::cerr << "No ROI found, skip preprocessing." << std::endl;
        image_and_ROI.set_preprocessed_ROI(preprocessed_ROI);
        return;
    }

    for (size_t i = 0; i < roi_image.size(); ++i) {
        if (roi_image[i].empty()) {
            std::cerr << "ROI " << i << " empty, skipped." << std::endl;
            continue;
        }

        // 1) BGR->GRAY (se serve)
        Mat gray;
        if (roi_image[i].channels() == 3)
            cvtColor(roi_image[i], gray, COLOR_BGR2GRAY);
        else
            gray = roi_image[i];

        // 2) Resize a 48x48 (come nel training)
        Mat resized;
        resize(gray, resized, Size(48, 48), 0, 0, INTER_LINEAR);

        // 3) float32 e normalizzazione [0,1]
        Mat f32;
        resized.convertTo(f32, CV_32F, 1.f / 255.f);

        // NB: 1 canale, 48x48, CV_32F
        preprocessed_ROI.push_back(f32);
    }

    image_and_ROI.set_preprocessed_ROI(preprocessed_ROI);
}
