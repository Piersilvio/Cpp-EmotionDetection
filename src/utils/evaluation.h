

#ifndef EVALUATION_H
#define EVALUATION_H



#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct DetectionEval {
    float precision = 0.f;
    float recall    = 0.f;
    int   tp        = 0;
    int   fp        = 0;
    int   fn        = 0;
    float mean_iou  = 0.f;
};

struct EmotionEval {
    int   correct  = 0;
    int   total    = 0;
    float accuracy = 0.f;
};

/**
 * Calcola le metriche di detection dati i bbox predetti e i bbox ground-truth.
 * Restituisce precision, recall, tp, fp, fn e mean IoU.
 */
DetectionEval evaluate_detection(const std::vector<cv::Rect>& predicted_faces,
                                 const std::vector<cv::Rect>& gt_boxes,
                                 float iou_threshold);

/**
 * Calcola le metriche di emotion recognition per i soli volti considerati TP
 * rispetto al ground-truth (best IoU > soglia). Confronta le etichette predette
 * con la label GT estratta dal nome file (helper già presenti in metrics.h).
 */
EmotionEval evaluate_emotions(const std::vector<std::string>& emotion_prediction,
                              const std::vector<cv::Rect>& predicted_faces,
                              const std::vector<cv::Rect>& gt_boxes,
                              float iou_threshold,
                              const std::string& image_file);



#endif //EVALUATION_H
