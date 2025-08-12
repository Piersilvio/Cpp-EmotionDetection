#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem> 
#include <vector>
#include <string>

#include "detection/detection.h"
#include "emotion_recognition/emotion_recognition.h"
#include "image/Image.h"

namespace fs = std::filesystem;
using namespace std;
using namespace cv;

const string FACE_DETECTOR_MODEL_PATH = "../models/haarcascade_frontalface_alt2.xml";
const string TENSORFLOW_MODEL_PATH = "../models/tensorflow_model.pb";

const string window_name = "Face detection and emotion recognition";

int main()
{
    const string images_folder = "../images/";

    vector<string> image_files;

    // Scan the folder with all the images
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

    // Print the list of all the images
    cout << "Scegli un'immagine da analizzare:" << endl;
    for (size_t i = 0; i < image_files.size(); i++) {
        cout << i << ": " << image_files[i] << endl;
    }

    // Let the user choose the image
    int choice = -1;
    while (choice < 0 || choice >= (int)image_files.size()) {
        cout << "Inserisci il numero corrispondente all'immagine: ";
        cin >> choice;
        if (cin.fail() || choice < 0 || choice >= (int)image_files.size()) {
            cout << "Scelta non valida, riprova." << endl;
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            choice = -1;
        }
    }

    Mat image = imread(image_files[choice]);
    if (image.empty()) {
        cout << "Errore nel caricamento dell'immagine." << endl;
        return -1;
    }

    namedWindow(window_name);

    Image image_and_ROI;

    // Detect all the faces with viola-jones
    detect_face(image);

    // Draw the bounding box
    image_and_ROI = draw_face_box(image);


    vector<Mat> roi_image = image_and_ROI.get_ROI();

    if (!roi_image.empty()) {
        // Preprocess ROI
        preprocessROI(roi_image, image_and_ROI);

        // Emotion prediction
        vector<Rect> detected_faces = get_detected_faces();
        vector<string> emotion_prediction = predict(image_and_ROI, TENSORFLOW_MODEL_PATH);

        // Print the label of the emotion
        image_and_ROI = print_predicted_label(image_and_ROI, emotion_prediction, detected_faces);
    }

    Mat output_image = image_and_ROI.get_pic();

    if (!output_image.empty()) {
        imshow(window_name, output_image);
    }
    else {
        imshow(window_name, image);
    }

    waitKey(0);

    return 0;
}
