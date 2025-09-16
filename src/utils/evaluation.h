//Tommaso Ballarin

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

/*
*   Calculate detection metrics given predicted bboxes and ground-truth bboxes.
*   Returns precision, recall, tp, fp, fn and mean IoU.
*/
DetectionEval evaluate_detection(const std::vector<cv::Rect>& predicted_faces,
                                 const std::vector<cv::Rect>& gt_boxes,
                                 float iou_threshold);

/*
*   Calculate emotion recognition metrics only for faces considered TP
*   compared to the ground-truth (best IoU > threshold). Compare the predicted labels
*   with the GT label extracted from the file name (helpers already present in metrics.h).
*/
EmotionEval evaluate_emotions(const std::vector<std::string>& emotion_prediction,
                              const std::vector<cv::Rect>& predicted_faces,
                              const std::vector<cv::Rect>& gt_boxes,
                              float iou_threshold,
                              const std::string& image_file);



#endif //EVALUATION_H
