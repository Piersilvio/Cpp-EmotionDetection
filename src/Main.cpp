#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>

#include "utils/filesystem.h"
#include "utils/selection.h"
#include "utils/pipeline.h"
#include "utils/config.h"

using namespace std;
using namespace cv;



/*
* Entry point of the program:
*   - Loads image paths
*   - Lets user select which images to process
*   - Processes selected images one by one
*   - Computes and prints global accuracy of emotion recognition
*/

int main() {
    const string images_folder = "../test/images/";
    const string labels_folder = "../test/labels/";

    vector<string> image_files = load_images(images_folder);

    if (image_files.empty()) {
        cout << "No images found in folder " << images_folder << endl;
        return -1;
    }

    vector<int> choices = select_images(image_files);

    namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);

    // reduce the window size
    cv::setWindowProperty(WINDOW_NAME, cv::WND_PROP_ASPECT_RATIO, cv::WINDOW_KEEPRATIO);
    cv::resizeWindow(WINDOW_NAME, 1280, 720);
    cv::moveWindow(WINDOW_NAME, 100, 100);

    int total_detected_faces = 0;
    int total_correct_emotions = 0;

    for (int choice : choices) {
        process_image(image_files[choice], labels_folder,
                      total_detected_faces, total_correct_emotions,
                      WINDOW_NAME);
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
