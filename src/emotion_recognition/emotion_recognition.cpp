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


std::vector<std::string> predict(Image& image, std::string model_filename) {

    // this takes the region of interest image and then runs model inference
    std::vector<cv::Mat> roi_image = image.getPreprocessedROI();
    
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


Image print_predicted_label(Image& image_and_ROI, std::vector<std::string>& emotion_prediction, std::vector<Rect> detected_faces) {

    Mat img = image_and_ROI.getPic();
    
    if (detected_faces.size() > 0) { 
        for (int i=0; i < detected_faces.size(); i++) {
            Rect r = detected_faces[i];

            // Write text prediction on bounding box
            putText(img, //target image
                        emotion_prediction[i], //text - will take the output of the model.inference()
                        Point(r.x, r.y-10), //top-left position of box
                        FONT_HERSHEY_DUPLEX,
                        1.0,
                        CV_RGB(118, 185, 0), //font color
                        2);
        }
    }

    image_and_ROI.setPic(img);

    return image_and_ROI;

}



