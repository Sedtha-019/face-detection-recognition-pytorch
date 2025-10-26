# Face Detection and Recognition using PyTorch

This project implements a real-time face detection and recognition system using PyTorch. The system can identify and label faces in both images and video streams.

##  Features

-  Face Detection using Faster R-CNN
-  Face Recognition using FaceNet (InceptionResNetV1 backbone)
-  Real-time detection and recognition using OpenCV
-  Structured training, validation, and test datasets
-  Modular notebooks and scripts for training and testing

---

---

##  How it Works

### 1. Face Detection (Faster R-CNN)
Trained on labeled face images using XML annotations. The detector identifies and returns bounding boxes of faces in an image or video.

### 2. Face Recognition (FaceNet)
The FaceNet model learns embeddings of faces, allowing it to match new faces with labeled identities from the training dataset.

---

##  Getting Started

###  Requirements

```bash
pip install torch torchvision facenet-pytorch opencv-python matplotlib Pillow lxml
```
 Real-Time Recognition
Use your webcam to detect and recognize faces in real-time:
python CV2+FasterRCNN.py      # Run face detection
python CV2+faceNet.py         # Run face recognition

 Training
➤ Train Faster R-CNN
Run the Jupyter notebook:

 Dataset
Images organized into folders per person (train/, valid/, test/)

Annotations for detection are in Pascal VOC XML format

Each identity has ~100 training images, 50 validation images

 License
This project is licensed under the MIT License. Feel free to use and modify it for your work.

 Acknowledgements
facenet-pytorch

PyTorch

LabelImg for annotation


---

