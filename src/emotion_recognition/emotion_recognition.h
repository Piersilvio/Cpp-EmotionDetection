#ifndef EMOTION_H
#define EMOTION_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include "../image/Image.h"

// Inference con modello ONNX (path passato come argomento)
std::vector<std::string> predict(Image& image, const std::string& model_filename);

// Preprocessing delle ROI per l’input del modello (GRAY 48x48 float32 [0,1])
void preprocessROI(std::vector<cv::Mat>& _roi_image, Image& image_and_ROI);

// Stampa le etichette predette
Image print_predicted_label(Image& image_and_ROI,
                            std::vector<std::string>& emotion_prediction,
                            std::vector<cv::Rect> detected_faces);

#endif
