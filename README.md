# Face Detection and Emotion Recognition

## Goal
The application takes static images in input. The aim is to detect a face for each person that is represented and, in addition, the interpretation of the facial expression emotion choosing between seven emotions: happy, sad, angry, disgust, surprise, fear, neutral. The detection is performed using the well-known Viola-Jones algorithm while the emotion recognition exploits a pre-trained Convolutional Neural
Network model.


## Brief technical description
App development has required the following main steps:

1. Train a neural network in Python with Keras/TensorFlow, starting from a GitHub notebook. Freeze the model and save both the graph and weights into a .pb file.
2. Perform Viola-Jones method with Haar Cascades to locate faces in the input image.
3. Enclose each detected face with a rectangular bounding box (region of interest).
4. Preprocess the region of interest to match the input expected by the model.
5. Use OpenCV’s Deep Neural Network (DNN) module to load the model and run inference on the detected faces.
6. Draw rectangular ground-truth boxes using the coordinates provided in test/labels subfolder.
7. Show the image on screen with bounding boxes and emotion predictions, while also printing evaluation metrics in the terminal.


## Minimum requirements
- cmake >= 3.22
- tensorflow <= 1.15
- compiler: g++ 11.4 (other versions untested)
- libraries: OpenCV 4.3.0 (other versions untested)


## Build and running instructions
1. In the top level directory of the repo make a build directory: 
   `mkdir build` 
   `cd build`
2. Compile: 
   `cmake ..`
   `make`
3. Run: `./FaceDetectionEmotionRecognition`.
