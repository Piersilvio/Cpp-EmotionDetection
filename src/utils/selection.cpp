//Tommaso Ballarin
#include "selection.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

vector<int> select_images(const vector<string>& image_files) {
    cout << "Select one or more images to analyze (e.g., 0 2 5):" << endl;
    for (size_t i = 0; i < image_files.size(); i++)
        cout << i << ": " << image_files[i] << endl;

    vector<int> choices;
    while (true) {
        cout << "Enter indices separated by space: ";
        string line;
        getline(cin, line);

        if (line.empty()) {
            cout << "No input entered. Try again." << endl;
            continue;
        }

        istringstream ss(line);
        int num;
        choices.clear();
        bool has_invalid = false;

        while (ss >> num) {
            if (num >= 0 && num < (int)image_files.size()) {
                choices.push_back(num);
            } else {
                cout << "Index " << num << " is invalid, ignored."  << endl;
                has_invalid = true;
            }
        }

        if (!choices.empty()) {
            if (has_invalid) cout << "Proceeding with valid indices entered." << endl;
            break;
        } else {
            cout <<" No valid indices entered. Try again." << endl;
        }
    }

    cout << "Selected indices: ";
    for (int i : choices) cout << i << " ";
    cout << endl;
    return choices;
}
