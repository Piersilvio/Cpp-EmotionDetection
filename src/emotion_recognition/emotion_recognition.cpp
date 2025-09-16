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


Image print_predicted_label(Image& image_and_ROI,
                            std::vector<std::string>& emotion_prediction,
                            std::vector<cv::Rect> detected_faces)
{
    cv::Mat img = image_and_ROI.get_pic();

#ifdef CV_AA
    const int lineType = CV_AA;      // OpenCV 2.x
#else
    const int lineType = cv::LINE_AA;
#endif

    const int boxThickness  = 2;
    const int textThickness = 1.25;
    const double fontScale  = 0.55;
    const int fontFace      = cv::FONT_HERSHEY_SIMPLEX;

    const size_t n = std::min(detected_faces.size(), emotion_prediction.size());
    for (size_t i = 0; i < n; ++i) {
        const cv::Rect& r = detected_faces[i];
        const cv::Scalar color = label_colors[i % label_colors.size()];

        // Disegna il riquadro
        cv::rectangle(img, r, color, boxThickness, lineType);

        // Testo in MAIUSCOLO
        std::string txt = emotion_prediction[i];
        std::transform(txt.begin(), txt.end(), txt.begin(), ::toupper);

        // --- centratura orizzontale del testo rispetto al riquadro ---
        int baseline = 0;
        cv::Size ts = cv::getTextSize(txt, fontFace, fontScale, textThickness, &baseline);

        int textX = r.x + (r.width - ts.width) / 2;  // centro orizzontale
        int textY = r.y - 5;                         // di default sopra al box

        // Se uscirebbe fuori in alto, spostalo dentro il box poco sotto il bordo superiore
        if (textY - ts.height < 0) {
            textY = r.y + ts.height + 5;
        }

        // Clamp orizzontale per sicurezza (evita di uscire dall'immagine)
        textX = std::max(0, std::min(textX, img.cols - ts.width));
        // Clamp verticale minimo (assicurati che la baseline sia visibile)
        textY = std::max(ts.height, std::min(textY, img.rows - 1));

        cv::putText(img, txt, cv::Point(textX, textY),
                    fontFace, fontScale, color, textThickness, lineType);
    }

    image_and_ROI.set_pic(img);
    return image_and_ROI;
}
