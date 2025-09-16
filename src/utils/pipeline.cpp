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

void process_image(const std::string& image_file, const std::string& labels_folder,
                   int& total_detected_faces, int& total_correct_emotions,
                   const std::string& window_name) {

    cv::Mat image = cv::imread(image_file);
    if (image.empty()) {
        std::cout << "Error loading image " << image_file << std::endl;
        return;
    }

    // Read ground truth
    std::string img_name = std::filesystem::path(image_file).stem().string();
    std::string label_file = labels_folder + "/" + img_name + ".txt";
    std::vector<cv::Rect> gt_boxes = read_ground_truth(label_file, image.cols, image.rows);
    draw_ground_truth(image, gt_boxes);

    std::vector<cv::Rect> ground_truth_faces = read_ground_truth(label_file, image.cols, image.rows);

    // Face detection
    std::vector<cv::Rect> detected_faces = detect_face(image, ground_truth_faces);

    // Estrazione ROI e (nel codice esistente) disegno box sull'immagine operativa
    Image image_and_ROI = draw_face_box(image);

    std::vector<cv::Mat> roi_image = image_and_ROI.get_ROI();
    std::vector<std::string> emotion_prediction;

    if (!roi_image.empty()) {
        preprocessROI(roi_image, image_and_ROI);
        emotion_prediction = predict(image_and_ROI, TENSORFLOW_MODEL_PATH);
        image_and_ROI = print_predicted_label(image_and_ROI, emotion_prediction, detected_faces);
    }

    // Detection evaluation
    std::vector<cv::Rect> predicted_faces = detected_faces;
    float iou_threshold = 0.45f;

    DetectionEval det = evaluate_detection(predicted_faces, gt_boxes, iou_threshold);

    std::cout << "Image: " << image_file << std::endl;
    std::cout << "TP: " << det.tp << ", FP: " << det.fp << ", FN: " << det.fn << std::endl;
    std::cout << "Precision: " << det.precision << ", Recall: " << det.recall
              << ", Mean IoU: " << det.mean_iou << std::endl;

    // Emotion evaluation
    if (!emotion_prediction.empty()) {
        std::string gt_label = extract_gt_label(image_file);
        std::string gt_norm = normalize_label(gt_label);

        int correct = 0;
        int total = 0;

        std::cout << "Predicted faces compared with GT (excluding FP):" << std::endl;

        for (size_t i = 0; i < predicted_faces.size(); i++) {
            float best_iou = 0.0f;
            for (const auto& g : gt_boxes) {
                float iou = IoU(predicted_faces[i], g);
                if (iou > best_iou) best_iou = iou;
            }

            if (best_iou > iou_threshold) {
                std::string pred_norm = normalize_label(clean_pred_label(emotion_prediction[i]));
                std::cout << "  Prediction: '" << pred_norm << "'  | GT: '" << gt_norm << "'" << std::endl;
                total++;
                total_detected_faces++;
                if (pred_norm == gt_norm) {
                    correct++;
                    total_correct_emotions++;
                }
            }
        }

        float emotion_accuracy = total > 0 ? static_cast<float>(correct) / total : 0.0f;
        std::cout << "Emotion recognition - correct: " << correct << "/" << total
                  << " (Accuracy: " << emotion_accuracy << ")" << std::endl;
    }

    cv::Mat output_image = image_and_ROI.get_pic();

    /* =====================  GUI compose (NEW)  ===================== */
    {
        const int WIN_W = 1280, WIN_H = 720;

        // Calcola la trasformazione (stessa logica di WINDOW_KEEPRATIO)
        const double sx = WIN_W / static_cast<double>(image.cols);
        const double sy = WIN_H / static_cast<double>(image.rows);
        const double scale = std::min(sx, sy);
        const int dispW = std::max(1, cvRound(image.cols * scale));
        const int dispH = std::max(1, cvRound(image.rows * scale));
        const int offX  = (WIN_W - dispW) / 2;
        const int offY  = (WIN_H - dispH) / 2;

        // Canvas pulito: rilegge l'immagine originale per evitare overlay preesistenti
        cv::Mat base = cv::imread(image_file);
        if (base.empty()) base = image.clone();
        cv::Mat vis(WIN_H, WIN_W, base.type(), cv::Scalar::all(0));

        const int interp = (scale < 1.0) ? cv::INTER_AREA : cv::INTER_LINEAR;
        cv::Mat resized;
        cv::resize(base, resized, cv::Size(dispW, dispH), 0, 0, interp);
        resized.copyTo(vis(cv::Rect(offX, offY, dispW, dispH)));

        // Mappa le bbox della detection alle coordinate finestra
        std::vector<cv::Rect> faces_gui = detected_faces;
        for (auto& r : faces_gui) {
            r.x      = cvRound(offX + r.x * scale);
            r.y      = cvRound(offY + r.y * scale);
            r.width  = cvRound(r.width  * scale);
            r.height = cvRound(r.height * scale);
        }

        // mappa le GT se vuoi visualizzarle
        std::vector<cv::Rect> gt_gui = gt_boxes;
        for (auto& r : gt_gui) {
            r.x      = cvRound(offX + r.x * scale);
            r.y      = cvRound(offY + r.y * scale);
            r.width  = cvRound(r.width  * scale);
            r.height = cvRound(r.height * scale);
        }

        // Disegno box + etichette nel canvas a risoluzione finestra
        #ifdef CV_AA
            const int lineType = CV_AA;      // OpenCV 2.x
        #else
            const int lineType = cv::LINE_AA;
        #endif

        draw_ground_truth(vis, gt_gui);    
    
        Image canvasWrap;
        canvasWrap.set_pic(vis); // il canvas diventa l'immagine interna all'oggetto

        // Disegna rettangoli + etichette con i "colori originali"
        canvasWrap = print_predicted_label(canvasWrap, emotion_prediction, faces_gui);

        // Recupera il Mat disegnato
        vis = canvasWrap.get_pic();
 

        // Da qui in poi usiamo 'vis' per salvare e mostrare
        output_image = vis;
    }
    /* ===================  end GUI compose (NEW)  =================== */

    // Saving of the image
    try {
        namespace fs = std::filesystem;
        fs::create_directories(OUTPUT_DIR); // if it does not exist, create

        const cv::Mat& annotated = output_image.empty() ? image : output_image; // fallback se qualcosa è andato storto
        std::string base = fs::path(image_file).stem().string();
        fs::path outPath = fs::path(OUTPUT_DIR) / (base + "_annotated.jpg");

        std::vector<int> jpgParams = { cv::IMWRITE_JPEG_QUALITY, 95 };
        if (cv::imwrite(outPath.string(), annotated, jpgParams)) {
            std::cout << "Saved annotated image: " << outPath.string() << std::endl;
        } else {
            std::cerr << "Failed to save image: " << outPath.string() << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Save error: " << e.what() << std::endl;
    }

    // Visualizzazione
    if (!output_image.empty()) cv::imshow(window_name, output_image);
    else                       cv::imshow(window_name, image);

    std::cout << "Press any key to proceed to the next image..." << std::endl;
    cv::waitKey(0);
}
