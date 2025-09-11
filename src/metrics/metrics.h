#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using std::vector;
using std::string;
using cv::Rect;

// Intersection over Union
float IoU(const cv::Rect& a, const cv::Rect& b);

// Precision e recall
void compute_metrics(const std::vector<cv::Rect>& predicted,
                     const std::vector<cv::Rect>& gt,
                     float iou_thresh,
                     float& precision,
                     float& recall,
                     int& tp,
                     int& fp,
                     int& fn);

vector<Rect> read_ground_truth(const string& label_file, int img_width, int img_height);


// Pulizia etichetta predetta: "sad : 68%" -> "sad"
string clean_pred_label(const string& raw_label);

// Estrae etichetta GT dal nome file: "sad(1).jpg" -> "sad"
string extract_gt_label(const string& filename);

// Normalizza stringa (minuscolo, senza spazi)
string normalize_label(const string& s);
