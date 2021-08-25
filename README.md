# Facial-Keypoint-Detection

[Deep Learning Nonodegree]

## Project Overview
In this project, I combined my knowledge of computer vision techniques and deep learning architectures to build a facial keypoint detection system. Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition, facial filters, and emotion recognition. The completed code is able to look at any image, detect faces, and predict the locations of facial keypoints on each face.

![Example](https://github.com/AhmedElgamiel/Facial-Keypoint-Detection/blob/main/key_pts_example.png)

## Implementation
1. Image pre-processing.
2. Detecting faces on the image with OpenCV Haar Cascades.
3. Detecting 68 facial keypoints with CNN with architecture of 5 (Conv layer + BatchNorm + Pooling layer + Dropout layer) and 3 Fully connected layers.
4. Putting all together to identify facial keypoints on any image.

## Software Stack
- Python
- Numpy
- Matplotlib
- CV2
- Pytourch
- Pandas
