#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "emotion_recognition.h"

using namespace cv;

// Ordine classi allineato al training FER2013
static const std::vector<std::string> EMO_LABELS = {
    "Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"
};

// Carica l'ONNX una sola volta (cache)
static cv::dnn::Net& getNet(const std::string& onnx_path) {
    static cv::dnn::Net net;
    static bool loaded = false;
    if (!loaded) {
        net = cv::dnn::readNetFromONNX(onnx_path);
        // Opzionale: abilita CUDA se disponibile
        // net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        // net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        loaded = true;
    }
    return net;
}

std::vector<std::string> predict(Image& image, const std::string& model_filename) {

    // ROI già preprocessate (48x48, GRAY, float32 [0,1]) dalla fase di preprocessing
    std::vector<cv::Mat> roi_image = image.get_preprocessed_ROI();
    std::vector<std::string> emotion_prediction;

    if (roi_image.empty()) return emotion_prediction;

    cv::dnn::Net& network = getNet(model_filename);

    for (const auto& roi : roi_image) {
        // blob NCHW: [1,1,48,48]. Scalefactor=1.0 perché ROI è già normalizzata.
        cv::Mat blob = cv::dnn::blobFromImage(
            roi, 1.0, cv::Size(48,48), cv::Scalar(),
            /*swapRB=*/false, /*crop=*/false, CV_32F
        );

        network.setInput(blob);
        cv::Mat prob = network.forward(); // [1,7] float32

        // Argmax
        cv::Point classId;
        double maxVal;
        cv::minMaxLoc(prob.reshape(1,1), nullptr, &maxVal, nullptr, &classId);
        int top_id = classId.x;
        float conf = static_cast<float>(maxVal);

        std::string res;
        if (top_id >= 0 && top_id < (int)EMO_LABELS.size())
            res = EMO_LABELS[top_id] + ": " + cv::format("%.2f%%", conf * 100.f);
        else
            res = "Unknown: 0%";

        emotion_prediction.push_back(res);
    }

    return emotion_prediction;
}

Image print_predicted_label(Image& image_and_ROI,
                            std::vector<std::string>& emotion_prediction,
                            std::vector<cv::Rect> detected_faces) {

    static std::vector<cv::Scalar> label_colors = {
        cv::Scalar(255, 0, 0),   // Blue
        cv::Scalar(0, 255, 0),   // Green
        cv::Scalar(0, 0, 255),   // Red
        cv::Scalar(255, 255, 0), // Cyan
        cv::Scalar(255, 0, 255), // Magenta
        cv::Scalar(0, 255, 255), // Yellow
        cv::Scalar(128, 0, 128)  // Purple
    };

    cv::Mat img = image_and_ROI.get_pic();

    if (!detected_faces.empty()) {
        for (size_t i = 0; i < detected_faces.size() && i < emotion_prediction.size(); ++i) {
            cv::Rect r = detected_faces[i];

            // Testo in maiuscolo
            std::string txt = emotion_prediction[i];
            std::transform(txt.begin(), txt.end(), txt.begin(), ::toupper);

            cv::putText(img,
                        txt,
                        cv::Point(r.x, r.y - 10),
                        cv::FONT_HERSHEY_COMPLEX,
                        1.0,
                        label_colors[i % label_colors.size()],
                        1);
        }
    }

    image_and_ROI.set_pic(img);
    return image_and_ROI;
}
