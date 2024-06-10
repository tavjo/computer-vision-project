# computer-vision-project

## Project Overview
This project aims to build a computer vision solution for counting and classifying certain objects from a dataset of images.

**Objective:** 
Develop a Python-based application that uses computer vision techniques to count and classify different objects from a set of images. The system should be able to differentiate at least three types of objects (e.g., cars, bicycles, pedestrians) in diverse lighting and background conditions. 

**Outline:**
1. Image Processing:
    - Pre-process images to enhance feature detection (e.g., noise reduction, contrast enhancement).

2. Object Detection and Classification:
    - Implement an object detection algorithm (you can use libraries such as OpenCV, TensorFlow, or PyTorch).
    - The system must differentiate between at least three different object types.
    - Each object detected should have a bounding box with a label displayed on the output image.

3.	Counting Logic:
    - Accurately count the number of each type of object within the image.
    - Display the counts on the output image.

4.	Output:
    - For each input image, generate an output image that shows bounding boxes, labels, and counts of each object type.
    - Print a summary report that includes total counts for each object type across all processed images.

## Getting Started

### Prerequisites
- Python 3.12.3
- Install dependencies:
  ```bash
  pip install -r requirements.txt

### Data Preparation

1. Download Data:
- Run the following script to download the dataset:

  ```bash
  python /scripts/download_data.py

2. Preprocess Data:
Use the provided scripts in src/data_preprocessing.py to preprocess the images.

### Training the Model

1. Train
- Run the following script to train the model:

  ```bash
  python src/train.py

### Prediction
