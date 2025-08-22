#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem> 
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include "detection/detection.h"
#include "emotion_recognition/emotion_recognition.h"
#include "image/Image.h"

namespace fs = std::filesystem;
using namespace std;
using namespace cv;

const string FACE_DETECTOR_MODEL_PATH = "../models/haarcascade_frontalface_alt2.xml";
const string TENSORFLOW_MODEL_PATH = "../models/tensorflow_model.pb";
const string window_name = "Face detection and emotion recognition";

cv::Mat resize_with_aspect_ratio(const cv::Mat& img, int max_width, int max_height) {
    int w = img.cols;
    int h = img.rows;
    float scale_w = (float)max_width / w;
    float scale_h = (float)max_height / h;
    float scale = std::min(scale_w, scale_h);  // usa il min per non superare limiti

    int new_w = static_cast<int>(w * scale);
    int new_h = static_cast<int>(h * scale);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));
    return resized;
}

// Funzione per leggere bounding box YOLO-style
vector<Rect> read_ground_truth(const string& label_file, int img_width, int img_height) {
    vector<Rect> boxes;
    ifstream infile(label_file);
    if (!infile.is_open()) return boxes;

    string line;
    while (getline(infile, line)) {
        istringstream ss(line);
        int class_id;
        float x_center, y_center, w, h;
        ss >> class_id >> x_center >> y_center >> w >> h;

        int x1 = static_cast<int>((x_center - w/2.0) * img_width);
        int y1 = static_cast<int>((y_center - h/2.0) * img_height);
        int width = static_cast<int>(w * img_width);
        int height = static_cast<int>(h * img_height);

        boxes.push_back(Rect(x1, y1, width, height));
    }
    return boxes;
}

// Disegna i bounding box ground truth
void draw_ground_truth(Mat& img, const vector<Rect>& boxes) {
    for (const auto& box : boxes) {
        rectangle(img, box, Scalar(0, 0, 255), 2); // rosso
    }
}

int main() {
    const string images_folder = "../test/images/";
    const string labels_folder = "../test/labels/";

    vector<string> image_files;

    for (const auto& entry : fs::directory_iterator(images_folder)) {
        if (entry.is_regular_file()) {
            string path = entry.path().string();
            if (path.find(".jpg") != string::npos || path.find(".png") != string::npos || path.find(".jpeg") != string::npos) {
                image_files.push_back(path);
            }
        }
    }

    if (image_files.empty()) {
        cout << "Nessuna immagine trovata nella cartella " << images_folder << endl;
        return -1;
    }

    cout << "Scegli una o più immagini da analizzare (es. 0 2 5):" << endl;
    for (size_t i = 0; i < image_files.size(); i++) {
        cout << i << ": " << image_files[i] << endl;
    }

    // Lettura indici multipli
    cin.ignore();
    string line;
    getline(cin, line);
    istringstream ss(line);
    vector<int> choices;
    int num;
    while (ss >> num) {
        if (num >= 0 && num < (int)image_files.size()) choices.push_back(num);
        else cout << "Indice " << num << " non valido, ignorato." << endl;
    }

    namedWindow(window_name);

    for (int choice : choices) {
        Mat image = imread(image_files[choice]);

        if (image.empty()) {
            cout << "Errore nel caricamento dell'immagine." << endl;
            continue;
        }

        // Ridimensiona a massimo 800x600 (puoi cambiare le dimensioni)
        image = resize_with_aspect_ratio(image, 800, 600);

        if (image.empty()) {
            cout << "Errore nel caricamento dell'immagine " << image_files[choice] << endl;
            continue;
        }

        // Leggi ground truth
        string img_name = fs::path(image_files[choice]).stem().string();
        string label_file = labels_folder + "/" + img_name + ".txt";
        vector<Rect> gt_boxes = read_ground_truth(label_file, image.cols, image.rows);
        draw_ground_truth(image, gt_boxes);

        // Face detection
        detect_face(image);
        Image image_and_ROI = draw_face_box(image);

        vector<Mat> roi_image = image_and_ROI.get_ROI();
        if (!roi_image.empty()) {
            preprocessROI(roi_image, image_and_ROI);
            vector<Rect> detected_faces = get_detected_faces();
            vector<string> emotion_prediction = predict(image_and_ROI, TENSORFLOW_MODEL_PATH);
            image_and_ROI = print_predicted_label(image_and_ROI, emotion_prediction, detected_faces);
        }

        Mat output_image = image_and_ROI.get_pic();
        if (!output_image.empty()) imshow(window_name, output_image);
        else imshow(window_name, image);

        cout << "Premi un tasto per passare all'immagine successiva..." << endl;
        waitKey(0);
    }

    return 0;
}
