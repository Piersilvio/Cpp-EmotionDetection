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
#include "metrics/metrics.h"


namespace fs = std::filesystem;
using namespace std;
using namespace cv;

const string FACE_DETECTOR_MODEL_PATH = "../models/haarcascade_frontalface_alt2.xml";
const string TENSORFLOW_MODEL_PATH = "../models/tensorflow_model.pb";
const string window_name = "Face detection and emotion recognition";




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

    // Stampa lista immagini
    cout << "Scegli una o più immagini da analizzare (es. 0 2 5):" << endl;
    for (size_t i = 0; i < image_files.size(); i++)
        cout << i << ": " << image_files[i] << endl;

    // Lettura indici robusta
    vector<int> choices;
    while (true) {
        cout << "Inserisci gli indici separati da spazio: ";
        string line;
        getline(cin, line);

        if (line.empty()) {
            cout << "Nessun input inserito. Riprova." << endl;
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
                cout << "Indice " << num << " non valido, ignorato." << endl;
                has_invalid = true;
            }
        }

        if (!choices.empty()) {
            if (has_invalid)
                cout << "Procedo con gli indici validi inseriti." << endl;
            break;
        } else {
            cout << "Nessun indice valido inserito. Riprova." << endl;
        }
    }

    cout << "Indici selezionati: ";
    for (int i : choices) cout << i << " ";
    cout << endl;

    namedWindow(window_name, cv::WINDOW_NORMAL);

    for (int choice : choices) {
        Mat image = imread(image_files[choice]);
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
        
       vector<Rect> predicted_faces = get_detected_faces();
        float precision = 0.0f, recall = 0.0f;
        int tp = 0, fp = 0, fn = 0;
        float iou_threshold = 0.5f;

        compute_metrics(predicted_faces, gt_boxes, iou_threshold, precision, recall, tp, fp, fn);

        // IoU media opzionale
        float mean_iou = 0.0f;
        int count = 0;
        for (const auto& p : predicted_faces) {
            for (const auto& g : gt_boxes) {
                float iou = IoU(p, g);
                if (iou > 0.0f) { mean_iou += iou; count++; }
            }
        }
        if (count > 0) mean_iou /= count;

        cout << "Immagine: " << image_files[choice] << endl;
        cout << "TP: " << tp << ", FP: " << fp << ", FN: " << fn << endl;
        cout << "Precision: " << precision << ", Recall: " << recall << ", Mean IoU: " << mean_iou << endl;



        Mat output_image = image_and_ROI.get_pic();
        if (!output_image.empty()) imshow(window_name, output_image);
        else imshow(window_name, image);

        cout << "Premi un tasto per passare all'immagine successiva..." << endl;
        waitKey(0);
    }

    return 0;
}
