

#ifndef PIPELINE_H
#define PIPELINE_H


#include <string>


/*
 *  Processes a single image:
 *      - Loads it and reads ground-truth bounding boxes
 *      - Runs face detection
 *      - Runs emotion recognition on detected faces
 *      - Computes detection metrics (precision, recall, IoU)
 *      - Compares predicted emotions with ground truth
 *      - Displays results on screen
 */

void process_image(const std::string& image_file, const std::string& labels_folder,
                   int& total_detected_faces, int& total_correct_emotions,
                   const std::string& window_name);




#endif //PIPELINE_H
