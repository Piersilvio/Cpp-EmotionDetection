//Tommaso Ballarin
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


std::vector<cv::Rect> map_bounding_boxes(std::vector<cv::Rect> boxes,
                                         int offX, int offY,
                                         double scale) {   
    for (auto& r : boxes) {
        r.x      = cvRound(offX + r.x * scale);
        r.y      = cvRound(offY + r.y * scale);
        r.width  = cvRound(r.width  * scale);
        r.height = cvRound(r.height * scale);
    }
    return boxes;
}


void process_image(const string& image_file, const string& labels_folder,
                   int& total_detected_faces, int& total_correct_emotions,
                   const string& window_name) {

    Mat image = imread(image_file);
    if (image.empty()) {
        cout << "Error loading image " << image_file << endl;
        return;
    }

    // Read ground truth
    string img_name = filesystem::path(image_file).stem().string();
    string label_file = labels_folder + "/" + img_name + ".txt";
    std::vector<cv::Rect> gt_boxes = read_ground_truth(label_file, image.cols, image.rows);
    draw_ground_truth(image, gt_boxes);

    std::vector<cv::Rect> ground_truth_faces = read_ground_truth(label_file, image.cols, image.rows);

    // Face detection
    std::vector<cv::Rect> detected_faces = detect_face(image, ground_truth_faces);

    // ROI extraction and (in the existing code) drawing boxes on the working image
    Image image_and_ROI = draw_face_box(image);

    vector<Mat> roi_image = image_and_ROI.get_ROI();
    vector<string> emotion_prediction;

    if (!roi_image.empty()) {
        preprocessROI(roi_image, image_and_ROI);
        emotion_prediction = predict(image_and_ROI, TENSORFLOW_MODEL_PATH);
        image_and_ROI = print_predicted_label(image_and_ROI, emotion_prediction, detected_faces);
    }

    // Detection evaluation
    std::vector<cv::Rect> predicted_faces = detected_faces;
    float iou_threshold = 0.45f;

    DetectionEval det = evaluate_detection(predicted_faces, gt_boxes, iou_threshold);

    cout << "\nImage: " << image_file << endl;
    cout << "TP: " << det.tp << ", FP: " << det.fp << ", FN: " << det.fn << endl;
    cout << "Precision: " << det.precision << ", Recall: " << det.recall
         << ", Mean IoU: " << det.mean_iou << endl;

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

        float emotion_accuracy = total > 0 ? static_cast<float>(correct) / total : 0.0f;
        cout << "Emotion recognition - correct: " << correct << "/" << total
             << " (Accuracy: " << emotion_accuracy << ")" << endl;
    }

    Mat output_image = image_and_ROI.get_pic();

    /* =====================  GUI compose (NEW)  ===================== */
    {
        const int WIN_W = 1280, WIN_H = 720;

        // Compute the transform (same logic as WINDOW_KEEPRATIO)
        const double sx = WIN_W / static_cast<double>(image.cols);
        const double sy = WIN_H / static_cast<double>(image.rows);
        const double scale = min(sx, sy);
        const int dispW = max(1, cvRound(image.cols * scale));
        const int dispH = max(1, cvRound(image.rows * scale));
        const int offX  = (WIN_W - dispW) / 2;
        const int offY  = (WIN_H - dispH) / 2;

        // Clean canvas: reload the original image to avoid pre-existing overlays
        Mat base = imread(image_file);
        if (base.empty()) base = image.clone();
        Mat vis(WIN_H, WIN_W, base.type(), Scalar::all(0));

        const int interp = (scale < 1.0) ? INTER_AREA : INTER_LINEAR;
        Mat resized;
        resize(base, resized, Size(dispW, dispH), 0, 0, interp);
        resized.copyTo(vis(Rect(offX, offY, dispW, dispH)));

        // Map faces bounding boxes to window coordinates
        std::vector<cv::Rect> faces_gui = detected_faces;
        faces_gui = map_bounding_boxes(faces_gui, offX, offY, scale);
       
        // Map GT boxes to window coordinates
        std::vector<cv::Rect> gt_gui = gt_boxes;
        gt_gui = map_bounding_boxes(gt_gui, offX, offY, scale);

        // Draw boxes + labels on the window-resolution canvas
        const int lineType = LINE_AA;

        draw_ground_truth(vis, gt_gui);    
    
        Image canvasWrap;
        canvasWrap.set_pic(vis); // the canvas becomes the internal image of the object

        // Draw rectangles + labels with the "original colors"
        canvasWrap = print_predicted_label(canvasWrap, emotion_prediction, faces_gui);

        // Retrieve the drawn Mat
        vis = canvasWrap.get_pic();
 
        // From here on we use 'vis' to save and display
        output_image = vis;
    }
    /* ===================  end GUI compose (NEW)  =================== */

    // Saving of the image
    try {
        namespace fs = filesystem;
        fs::create_directories(OUTPUT_DIR); // if it does not exist, create

        const Mat& annotated = output_image.empty() ? image : output_image; // fallback if something went wrong
        string base = fs::path(image_file).stem().string();
        fs::path outPath = fs::path(OUTPUT_DIR) / (base + "_annotated.jpg");

        vector<int> jpgParams = { IMWRITE_JPEG_QUALITY, 95 };
        if (imwrite(outPath.string(), annotated, jpgParams)) {
            cout << "Saved annotated image: " << outPath.string() << endl;
        } else {
            cerr << "Failed to save image: " << outPath.string() << endl;
        }

    } catch (const exception& e) {
        cerr << "Save error: " << e.what() << endl;
    }

    // Display
    if (!output_image.empty()) imshow(window_name, output_image);
    else                       imshow(window_name, image);

    cout << "Press any key to proceed to the next image..." << endl;
    waitKey(0);
}

