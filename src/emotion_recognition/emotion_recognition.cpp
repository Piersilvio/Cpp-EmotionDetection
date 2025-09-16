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
Scalar(0, 0, 255),     // Red
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


Image print_predicted_label(Image& image_and_ROI, std::vector<std::string>& emotion_prediction, std::vector<Rect> detected_faces) {

    Mat img = image_and_ROI.get_pic();
    
    if (detected_faces.size() > 0) { 
        for (int i=0; i < detected_faces.size(); i++) {
            Rect r = detected_faces[i];

              // Convert to upper case
              std::string emotion_upper = emotion_prediction[i];
              std::transform(emotion_upper.begin(), emotion_upper.end(), emotion_upper.begin(), ::toupper);
          
              cv::putText(img,
                          emotion_upper,
                          cv::Point(r.x, r.y - 10),
                          cv::FONT_HERSHEY_COMPLEX,
                          1.0,                          // scale
                          label_colors[i % label_colors.size()],    // same color of the bounding box
                          1);   
        }
    }

    image_and_ROI.set_pic(img);

    return image_and_ROI;

}



