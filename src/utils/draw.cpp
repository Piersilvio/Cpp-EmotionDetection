//Spicoli Piersilvio

#include "draw.h"
using namespace cv;
using namespace std;

// Draws the ground-truth bounding boxes (red rectangles) on an image.
void draw_ground_truth(Mat& img, const vector<Rect>& boxes) {
    for (const auto& box : boxes) {
        rectangle(img, box, Scalar(0, 0, 255), 2);
    }
}
