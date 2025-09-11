a#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem> 
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>

#include "detection/detection.h"
#include "emotion_recognition/emotion_recognition.h"
#include "image/Image.h"
#include "metrics/metrics.h"

namespace fs = std::filesystem;
using namespace std;
using namespace cv;

const string FACE_DETECTOR_MODEL_PATH = "../models/haarcascade_frontalface_alt2.xml";
const string TENSORFLOW_MODEL_PATH = "../models/tensorflow_model.pb";
const string window_name = "Face detection and emotion recognition";

// Draws the ground-truth bounding boxes (red rectangles) on an image.
void draw_ground_truth(Mat& img, const vector<Rect>& boxes) {
    for (const auto& box : boxes) {
        rectangle(img, box, Scalar(0, 0, 255), 2);
    }
}

// Loads all image file paths from a given folder and
// returns a vector of file paths.
vector<string> load_images(const string& folder) {
    vector<string> image_files;
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (entry.is_regular_file()) {
            string path = entry.path().string();
            if (path.find(".jpg") != string::npos || path.find(".png") != string::npos || path.find(".jpeg") != string::npos) {
                image_files.push_back(path);
            }
        }
    }
    return image_files;
}

// Lets the user select which images to process by typing their indices.
// Returns a vector of chosen indices.
vector<int> select_images(const vector<string>& image_files) {
    cout << "Select one or more images to analyze (e.g., 0 2 5):" << endl;
    for (size_t i = 0; i < image_files.size(); i++)
        cout << i << ": " << image_files[i] << endl;

    vector<int> choices;
    while (true) {
        cout << "Enter indices separated by space: ";
        string line;
        getline(cin, line);

        if (line.empty()) {
            cout << "No input entered. Try again." << endl;
            continue;
        }

        istringstream ss(line);
        int num;
        choices.clear();
        bool has_invalid = false;

        while (ss >> num) {
            if (num >= 0 && num < (int)image_files.size()) {
                choices.push_back(num);
            } else {
                cout << "Index " << num << " is invalid, ignored."  << endl;
                has_invalid = true;
            }
        }

        if (!choices.empty()) {
            if (has_invalid) cout << "Proceeding with valid indices entered." << endl;
            break;
        } else {
            cout << No valid indices entered. Try again." << endl;
        }
    }

    cout << "Selected indices: ";
    for (int i : choices) cout << i << " ";
    cout << endl;
    return choices;
}

//// Processes a single image:
// - Loads it and reads ground-truth bounding boxes
// - Runs face detection
// - Runs emotion recognition on detected faces
// - Computes detection metrics (precision, recall, IoU)
// - Compares predicted emotions with ground truth
// - Displays results on screen
void process_image(const string& image_file, const string& labels_folder,
                   int& total_detected_faces, int& total_correct_emotions) {

    Mat image = imread(image_file);
    if (image.empty()) {
        cout << "Error loading image " << image_file << endl;
        return;
    }

    // Read ground truth
    string img_name = fs::path(image_file).stem().string();
    string label_file = labels_folder + "/" + img_name + ".txt";
    vector<Rect> gt_boxes = read_ground_truth(label_file, image.cols, image.rows);
    draw_ground_truth(image, gt_boxes);
    
    std::vector<cv::Rect> ground_truth_faces = read_ground_truth(label_file, image.cols, image.rows); 
    
    // Face detection
    detect_face(image,ground_truth_faces);
    
    Image image_and_ROI = draw_face_box(image);

    vector<Mat> roi_image = image_and_ROI.get_ROI();
    vector<string> emotion_prediction;

    if (!roi_image.empty()) {
        preprocessROI(roi_image, image_and_ROI);
        vector<Rect> detected_faces = get_detected_faces();
        emotion_prediction = predict(image_and_ROI, TENSORFLOW_MODEL_PATH);
        image_and_ROI = print_predicted_label(image_and_ROI, emotion_prediction, detected_faces);
    }

    // Detection evaluation
    vector<Rect> predicted_faces = get_detected_faces();
    float precision = 0.0f, recall = 0.0f;
    int tp = 0, fp = 0, fn = 0;
    float iou_threshold = 0.3f;

    compute_metrics(predicted_faces, gt_boxes, iou_threshold, precision, recall, tp, fp, fn);

    float mean_iou = 0.0f;
    int count = 0;
    for (const auto& p : predicted_faces) {
        for (const auto& g : gt_boxes) {
            float iou = IoU(p, g);
            if (iou > 0.0f) { mean_iou += iou; count++; }
        }
    }
    if (count > 0) mean_iou /= count;

    cout << "Image: " << image_file << endl;
    cout << "TP: " << tp << ", FP: " << fp << ", FN: " << fn << endl;
    cout << "Precision: " << precision << ", Recall: " << recall << ", Mean IoU: " << mean_iou << endl;

    // Emotion evaluation
    if (!emotion_prediction.empty()) {
        string gt_label = extract_gt_label(image_file);
        string gt_norm = normalize_label(gt_label);

        int correct = 0;
        int total = 0;

        cout << "Predicted faces compared with GT (excluding FP):" << endl;

        for (size_t i = 0; i < predicted_faces.size(); i++) {
            float best_iou = 0.0f;
            for (const auto& g : gt_boxes) {
                float iou = IoU(predicted_faces[i], g);
                if (iou > best_iou) best_iou = iou;
            }

            if (best_iou > iou_threshold) {
                string pred_norm = normalize_label(clean_pred_label(emotion_prediction[i]));
                cout << "  Prediction: '" << pred_norm << "'  | GT: '" << gt_norm << "'" << endl;
                total++;
                total_detected_faces++;
                if (pred_norm == gt_norm) {
                    correct++;
                    total_correct_emotions++;
                }
            }
        }

        float emotion_accuracy = total > 0 ? (float)correct / total : 0.0f;
        cout << "Emotion recognition - correct: " << correct << "/" << total
             << " (Accuracy: " << emotion_accuracy << ")" << endl;
    }

    Mat output_image = image_and_ROI.get_pic();
    if (!output_image.empty()) imshow(window_name, output_image);
    else imshow(window_name, image);

    cout << "Press any key to proceed to the next image..." << endl;
    waitKey(0);
}

// Entry point of the program:
// - Loads image paths
// - Lets user select which images to process
// - Processes selected images one by one
// - Computes and prints global accuracy of emotion recognition
int main() {
    const string images_folder = "../test/images/";
    const string labels_folder = "../test/labels/";

    vector<string> image_files = load_images(images_folder);

    if (image_files.empty()) {
        cout << "No images found in folder " << images_folder << endl;
        return -1;
    }

    vector<int> choices = select_images(image_files);

    namedWindow(window_name, cv::WINDOW_NORMAL);

    int total_detected_faces = 0;
    int total_correct_emotions = 0;

    for (int choice : choices) {
        process_image(image_files[choice], labels_folder, total_detected_faces, total_correct_emotions);
    }

    if (total_detected_faces > 0) {
        float global_accuracy = (float)total_correct_emotions / total_detected_faces;
        cout << "\n=== Global statistics for selected images ===" << endl;
        cout << "Faces correctly detected with correct emotion: "
             << total_correct_emotions << "/" << total_detected_faces
             << " (Global accuracy: " << global_accuracy << ")" << endl;
    } else {
        cout << "No faces correctly detected in the selected images" << endl;
    }

    return 0;
}
