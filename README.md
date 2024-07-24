# Real-Time Face Recognition Project

## Overview
This project demonstrates real-time face detection, data collection, and face recognition using OpenCV and K-Nearest Neighbors (KNN). It captures video frames from a webcam, detects faces using the Haar Cascade Classifier, collects face data, and recognizes faces in real-time.

## Project Structure
- **video_read.py**: Captures and displays video frames from a webcam.
- **face_detection.py**: Detects faces in video frames using the Haar Cascade Classifier and displays the detected faces.
- **face_data.py**: Collects face data from video frames and stores it in a dataset.
- **face_recognition.py**: Recognizes faces in real-time using the KNN algorithm.

## Setup and Installation
1. **Clone the Repository**:
    ```sh
    git clone https://github.com/yourusername/real-time-face-recognition.git
    cd real-time-face-recognition
    ```

2. **Install Dependencies**:
    Make sure you have Python installed. Then, install the required libraries:
    ```sh
    pip install numpy opencv-python
    ```

3. **Download Haar Cascade Classifier**:
    Download the `haarcascade_frontalface_alt.xml` file and place it in the project directory.

## Running the Project
1. **Video Capture**:
    ```sh
    python video_read.py
    ```

2. **Face Detection**:
    ```sh
    python face_detection.py
    ```

3. **Face Data Collection**:
    ```sh
    python face_data.py
    ```
    Enter the name of the person when prompted. This will save the face data in the `face_dataset` directory.

4. **Face Recognition**:
    ```sh
    python face_recognition.py
    ```

## Key Concepts and Theory
### Haar Cascade Classifier
Developed by Paul Viola and Michael Jones, the Haar Cascade Classifier is used for object detection and is particularly effective for face detection. It works by training on a large number of positive and negative images, learning to identify features that distinguish faces from non-faces.

### K-Nearest Neighbors (KNN)
KNN is a simple, instance-based learning algorithm used for classification and regression. It classifies new data points based on the majority label of the 'k' nearest neighbors from the training dataset. The algorithm is easy to implement and interpret, making it ideal for face recognition tasks.

## Applications
- **Security Systems**: Enhance security with real-time face recognition.
- **Authentication**: Securely unlock devices or verify identities.
- **Personalized Experience**: Provide tailored services in retail or hospitality.

## Advantages
- **Real-Time Processing**: Immediate face detection and recognition.
- **High Accuracy**: Robust face detection and recognition.
- **Scalability**: Expand the dataset to recognize multiple faces.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt) file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## Contact
For any questions or collaboration ideas, feel free to reach out email - satvik.shankar2003@gmail.com !

---

Satvik Shankar

