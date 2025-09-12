#include "evaluation.h"
#include "../metrics/metrics.h"
#include <algorithm>


using namespace cv;
using namespace std;

static float meanIoU(const std::vector<cv::Rect>& predicted,
                     const std::vector<cv::Rect>& gt)
{
    if (predicted.empty() || gt.empty()) return 0.f;

    float sum = 0.f;
    int   cnt = 0;
    for (const auto& p : predicted) {
        for (const auto& g : gt) {
            float i = IoU(p, g);
            if (i > 0.f) { sum += i; ++cnt; }
        }
    }
    return (cnt > 0) ? (sum / cnt) : 0.f;
}

DetectionEval evaluate_detection(const std::vector<cv::Rect>& predicted_faces,
                                 const std::vector<cv::Rect>& gt_boxes,
                                 float iou_threshold)
{
    DetectionEval out{};
    float precision = 0.f, recall = 0.f;
    int tp = 0, fp = 0, fn = 0;

    compute_metrics(predicted_faces, gt_boxes, iou_threshold,
                    precision, recall, tp, fp, fn);

    out.precision = precision;
    out.recall    = recall;
    out.tp = tp; out.fp = fp; out.fn = fn;
    out.mean_iou  = meanIoU(predicted_faces, gt_boxes);
    return out;
}

EmotionEval evaluate_emotions(const std::vector<std::string>& emotion_prediction,
                              const std::vector<cv::Rect>& predicted_faces,
                              const std::vector<cv::Rect>& gt_boxes,
                              float iou_threshold,
                              const std::string& image_file)
{
    EmotionEval out{};

    if (emotion_prediction.empty() || predicted_faces.empty())
        return out;

    std::string gt_label = extract_gt_label(image_file);
    std::string gt_norm  = normalize_label(gt_label);

    for (size_t i = 0; i < predicted_faces.size() && i < emotion_prediction.size(); ++i) {
        float best_iou = 0.0f;
        for (const auto& g : gt_boxes)
            best_iou = std::max(best_iou, IoU(predicted_faces[i], g));

        if (best_iou > iou_threshold) {
            std::string pred_norm = normalize_label(clean_pred_label(emotion_prediction[i]));
            out.total++;
            if (pred_norm == gt_norm) out.correct++;
        }
    }

    out.accuracy = (out.total > 0) ? (static_cast<float>(out.correct) / out.total) : 0.f;
    return out;
}

