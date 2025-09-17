//Tommaso Ballarin

#ifndef PIPELINE_H
#define PIPELINE_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector> 

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

// Map bounding boxes to window coordinates
std::vector<cv::Rect> map_bounding_boxes(std::vector<cv::Rect> boxes,
                                         int offX, int offY,
                                         double scale);


#endif //PIPELINE_H
