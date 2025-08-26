#include <opencv2/opencv.hpp>

#include "emotion_recognition.h"

using namespace cv;


// Mapping of the class id to the string label
std::map<int, std::string> classid_to_string = {
                         {0, "Angry"}, 
                         {1, "Disgust"}, 
                         {2, "Fear"}, 
                         {3, "Happy"}, 
                         {4, "Sad"}, 
                         {5, "Surprise"}, 
                         {6, "Neutral"}} ;  // Create a map from class id to the class labels
                         
// Label colors
std::vector<cv::Scalar> label_colors = {
cv::Scalar(255, 0, 0),     // Blue
cv::Scalar(0, 255, 0),     // Green
cv::Scalar(0, 0, 255),     // Red
cv::Scalar(255, 255, 0),   // Cyan
cv::Scalar(255, 0, 255),   // Magenta
cv::Scalar(0, 255, 255),   // Yellow
cv::Scalar(128, 0, 128)    // Purple
};


/*
std::vector<std::string> predict(Image& image, std::string model_filename) {

    // this takes the region of interest image and then runs model inference
    std::vector<cv::Mat> roi_image = image.get_preprocessed_ROI();
    
    cv::dnn::Net network = dnn::readNet(model_filename); // Load the tensorflow model 
    std::vector<std::string> emotion_prediction;

    if (roi_image.size() > 0) { 
        for (int i=0; i < roi_image.size(); i++) {
            // Convert to blob
            Mat blob = dnn::blobFromImage(roi_image[i]);

            // Pass blob to network
            network.setInput(blob);

            // Forward pass on network    
            Mat prob = network.forward();

            // Sort the probabilities and rank the indicies
            Mat sorted_probabilities;
            Mat sorted_ids;
            sort(prob.reshape(1, 1), sorted_probabilities, SORT_DESCENDING);
            sortIdx(prob.reshape(1, 1), sorted_ids, SORT_DESCENDING);

            // Get top probability and top class id
            float top_probability = sorted_probabilities.at<float>(0);
            int top_class_id = sorted_ids.at<int>(0);

            // Map classId to the class name string (ie. happy, sad, angry, disgust etc.)
            std::string class_name = classid_to_string.at(top_class_id);

            // Prediction result string to print
            std::string result_string = class_name + ": " + std::to_string(top_probability * 100) + "%";

            // Put on end of result vector
            emotion_prediction.push_back(result_string);

        }
    }

    return emotion_prediction;

}
*/


std::vector<std::string> predict(Image& image, std::string model_filename) {

    // Get the preprocessed regions of interest (ROIs) from the image
    std::vector<cv::Mat> roi_images = image.get_preprocessed_ROI();
    
    // Load the ONNX model
    cv::dnn::Net network = cv::dnn::readNetFromONNX(model_filename);
    
    // Aggiungi un controllo di errore per il caricamento del modello
    if (network.empty()) {
        throw std::runtime_error("Impossibile caricare il modello ONNX da: " + model_filename);
    }

    std::vector<std::string> emotion_prediction;

    // Check if there are any ROIs
    if (!roi_images.empty()) {
        for (const auto& roi : roi_images) {

            // Preprocess the ROI for the network
            cv::Mat blob = cv::dnn::blobFromImage(roi, 1.0, cv::Size(224, 224), cv::Scalar(), true, false, CV_32F);

            // Set the blob as network input
            network.setInput(blob);

            // Perform forward pass to get network output
            cv::Mat prob = network.forward();

            // Use minMaxLoc to find the top prediction directly
            double top_probability;
            cv::Point class_id_point;
            
            // Trova la posizione e il valore del massimo
            cv::minMaxLoc(prob, nullptr, &top_probability, nullptr, &class_id_point);
            
            int top_class_id = class_id_point.x;
            
            // Map the class ID to the emotion label
            if (classid_to_string.count(top_class_id)) {
                std::string class_name = classid_to_string.at(top_class_id);

                // Create the prediction string
                std::string result_string = class_name + ": " + std::to_string(top_probability * 100) + "%";

                // Add the prediction to the results vector
                emotion_prediction.push_back(result_string);
            } else {
                 emotion_prediction.push_back("Classe non trovata per ID: " + std::to_string(top_class_id));
            }
        }
    }

    // Return all predictions
    return emotion_prediction;
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



