
#ifndef FILESYSTEM_H
#define FILESYSTEM_H

#include <string>
#include <vector>


/*
 *  Loads all image file paths from a given folder and
 *  returns a vector of file paths.
 */

std::vector<std::string> load_images(const std::string& folder);



#endif //FILESYSTEM_H
