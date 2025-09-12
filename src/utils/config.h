

#ifndef CONFIG_H
#define CONFIG_H

#include <string>

inline const std::string FACE_DETECTOR_MODEL_PATH = "../models/haarcascade_frontalface_alt2.xml";
inline const std::string TENSORFLOW_MODEL_PATH    = "../models/fer2013_cnn.onnx";
inline const std::string WINDOW_NAME              = "Face detection and emotion recognition";

inline const std::string OUTPUT_DIR               = "../output";   // cambia se vuoi

#endif //CONFIG_H
