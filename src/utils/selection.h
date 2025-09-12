//
// Created by pspic on 11/09/2025.
//

#ifndef SELECTION_H
#define SELECTION_H



#include <string>
#include <vector>

// Lets the user select which images to process by typing their indices.
// Returns a vector of chosen indices.
std::vector<int> select_images(const std::vector<std::string>& image_files);


#endif //SELECTION_H
