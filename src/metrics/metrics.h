//Spicoli Piersilvio

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using std::vector;
using std::string;
using cv::Rect;

// Intersection over Union
float IoU(const cv::Rect& a, const cv::Rect& b);

// Precision, recall, FP, TP, FN
void compute_metrics(const std::vector<cv::Rect>& predicted,
                     const std::vector<cv::Rect>& gt,
                     float iou_thresh,
                     float& precision,
                     float& recall,
                     int& tp,
                     int& fp,
                     int& fn);

vector<Rect> read_ground_truth(const string& label_file, int img_width, int img_height);


//Clean label
string clean_pred_label(const string& raw_label);

//Extract label
string extract_gt_label(const string& filename);

//Label normalization
string normalize_label(const string& s);
