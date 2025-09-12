#include "pipeline.h"

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <filesystem>

#include "../detection/detection.h"
#include "../emotion_recognition/emotion_recognition.h"
#include "../image/Image.h"
#include "../metrics/metrics.h"
#include "draw.h"
#include "evaluation.h"
#include "config.h"

using namespace std;
using namespace cv;

void process_image(const string& image_file, const string& labels_folder,
                   int& total_detected_faces, int& total_correct_emotions,
                   const string& window_name) {

    Mat image = imread(image_file);
    if (image.empty()) {
        cout << "Error loading image " << image_file << endl;
        return;
    }

    // Read ground truth
    string img_name = std::filesystem::path(image_file).stem().string();
    string label_file = labels_folder + "/" + img_name + ".txt";
    vector<Rect> gt_boxes = read_ground_truth(label_file, image.cols, image.rows);
    draw_ground_truth(image, gt_boxes);

    std::vector<cv::Rect> ground_truth_faces = read_ground_truth(label_file, image.cols, image.rows);

    // Face detection
    detect_face(image, ground_truth_faces);

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
    float iou_threshold = 0.3f;

    DetectionEval det = evaluate_detection(predicted_faces, gt_boxes, iou_threshold);

    cout << "Image: " << image_file << endl;
    cout << "TP: " << det.tp << ", FP: " << det.fp << ", FN: " << det.fn << endl;
    cout << "Precision: " << det.precision << ", Recall: " << det.recall
         << ", Mean IoU: " << det.mean_iou << endl;

    // Emotion evaluation
    if (!emotion_prediction.empty()) {
        EmotionEval emo = evaluate_emotions(emotion_prediction, predicted_faces, gt_boxes,
                                            iou_threshold, image_file);

        cout << "Predicted faces compared with GT (excluding FP):" << endl;
        cout << "Emotion recognition - correct: " << emo.correct << "/" << emo.total
             << " (Accuracy: " << emo.accuracy << ")" << endl;

        total_detected_faces   += emo.total;
        total_correct_emotions += emo.correct;
    }

    Mat output_image = image_and_ROI.get_pic();

    // saving annoted image
    try {
        namespace fs = std::filesystem;
        fs::create_directories(OUTPUT_DIR); // create the "output" dir to store

        const Mat& annotated = output_image.empty() ? image : output_image;
        string base = fs::path(image_file).stem().string();
        fs::path outPath = fs::path(OUTPUT_DIR) / (base + "_annotated.jpg");

        vector<int> jpgParams = { cv::IMWRITE_JPEG_QUALITY, 95 };
        if (cv::imwrite(outPath.string(), annotated, jpgParams)) {
            cout << "Saved annotated image: " << outPath.string() << endl;
        } else {
            cerr << "Failed to save image: " << outPath.string() << endl;
        }

    } catch (const std::exception& e) {
        cerr << "Save error: " << e.what() << endl;
    }

    // ------------------------------------------------------------------------------

    if (!output_image.empty()) imshow(window_name, output_image);
    else imshow(window_name, image);

    cout << "Press any key to proceed to the next image..." << endl;
    waitKey(0);
}
