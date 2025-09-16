//Tommaso Ballarin
#include "filesystem.h"
#include <filesystem>
#include <vector>
#include <string>

namespace fs = std::filesystem;
using namespace std;

vector<string> load_images(const string& folder) {
    vector<string> image_files;
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (entry.is_regular_file()) {
            string path = entry.path().string();
            if (path.find(".jpg") != string::npos || path.find(".png") != string::npos || path.find(".jpeg") != string::npos) {
                image_files.push_back(path);
            }
        }
    }
    return image_files;
}
