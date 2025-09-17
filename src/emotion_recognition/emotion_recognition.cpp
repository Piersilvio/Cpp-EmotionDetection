// Author: Jacopo Onorati

#include <opencv2/opencv.hpp>

#include "emotion_recognition.h"

using namespace cv;
using namespace std;


// Mapping of the class id to the string label
map<int, string> classid_to_string = {
                         {0, "Angry"}, 
                         {1, "Disgust"}, 
                         {2, "Fear"}, 
                         {3, "Happy"}, 
                         {4, "Sad"}, 
                         {5, "Surprise"}, 
                         {6, "Neutral"}} ;
                         
//Label colors
vector<Scalar> label_colors = {
Scalar(255, 0, 0),     // Blue
Scalar(0, 255, 0),     // Green
Scalar(0, 0, 140),     // modified Red
Scalar(255, 255, 0),   // Cyan
Scalar(255, 0, 255),   // Magenta
Scalar(0, 255, 255),   // Yellow
Scalar(128, 0, 128)    // Purple
};



vector<string> predict(Image& img, string model) {

    vector<Mat> prep_ROI= img.get_preprocessed_ROI();
    
    // Load model 
    dnn::Net network = dnn::readNet(model); 
    
    // Vector that will contain the emotion prediction for each input ROI
    vector<string> predictions;

    if (prep_ROI.size() > 0) { 
        for (int i=0; i < prep_ROI.size(); i++) {
            // Convert to blob
            Mat blob = dnn::blobFromImage(prep_ROI[i]);

            // Pass blob to network
            network.setInput(blob);

            // Forward pass on network    
            Mat prob = network.forward();

            // Sort the probabilities and rank the indices
            Mat sorted_probabilities;
            Mat sorted_ids;
            cv::sort(prob.reshape(1, 1), sorted_probabilities, SORT_DESCENDING);
            cv::sortIdx(prob.reshape(1, 1), sorted_ids, SORT_DESCENDING);

            // Get top probability and top class id
            float top_probability = sorted_probabilities.at<float>(0);
            int top_class_id = sorted_ids.at<int>(0);

            string class_name = classid_to_string.at(top_class_id);

            // Prediction result string to print
            string result_string = class_name + ": " + to_string(top_probability * 100) + "%";

            predictions.push_back(result_string);

        }
    }

    return predictions;

}


Image print_predicted_label(Image& image_and_ROI,
                            vector<string>& emotion_prediction,
                            vector<Rect> detected_faces)
{
    Mat img = image_and_ROI.get_pic();

    const int type_of_line = LINE_AA;

    const int box_thickness  = 2;
    const int text_thickness = 1.25;
    const double font_scale  = 0.55;
    const int font_face      = cv::FONT_HERSHEY_SIMPLEX;

    const size_t n = min(detected_faces.size(), emotion_prediction.size());
    for (size_t i = 0; i < n; ++i) {
        const Rect& r = detected_faces[i];
        const Scalar color = label_colors[i % label_colors.size()];

        // Draw the bounding box
        rectangle(img, r, color, box_thickness, type_of_line);

        // Text in UPPERCASE
        string txt = emotion_prediction[i];
        transform(txt.begin(), txt.end(), txt.begin(), ::toupper);

        // Horizontal centering of the text with respect to the box
        int baseline = 0;
        Size ts = getTextSize(txt, font_face, font_scale, text_thickness, &baseline);

        int textX = r.x + (r.width - ts.width) / 2;  // horizontal center
        int textY = r.y - 5;                         // by default above the box

        // If it would go out on top, move it inside the box just below the top edge
        if (textY - ts.height < 0) {
            textY = r.y + ts.height + 5;
        }

        // Horizontal clamp for safety (prevents going outside the image)
        textX = max(0, min(textX, img.cols - ts.width));
        // Minimum vertical clamp (ensures the baseline is visible)
        textY = max(ts.height, min(textY, img.rows - 1));

        putText(img, txt, Point(textX, textY),
                    font_face, font_scale, color, text_thickness, type_of_line);
    }

    image_and_ROI.set_pic(img);
    return image_and_ROI;
}

