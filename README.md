# ECG-Based Human Activity Classification

## Overview

This project classifies human activities (e.g., sitting, walking, jogging, solving math problems) using ECG (Electrocardiogram) signal data. It uses both machine learning and deep learning approaches.

## Project Structure

1. **Feature-Based Classification**:

   * Extracts features like heart rate, RR intervals, skewness, and kurtosis from ECG signals.
   * Uses Random Forest Classifier for activity classification.

2. **Image-Based Classification**:

   * Converts ECG signals into scalogram images using wavelet transforms.
   * Trains a CNN model (MobileNetV2) on these images.

## Activities

The following activities are classified:

* Sitting
* Walking
* Jogging
* Hand biking
* Solving math problems

## Dataset

* ECG signals are collected from multiple subjects.
* Each subject performs all 5 activities.
* Signals are stored in `.tsv` format under folders for each subject and activity.

## Tools and Libraries

* Python
* Pandas, NumPy, Matplotlib
* NeuroKit2
* Scikit-learn
* TensorFlow / Keras
* PyWavelets

## Steps to Run

1. Load ECG data and extract features.
2. Train a Random Forest Classifier on the features.
3. Generate scalogram images from ECG signals.
4. Train a CNN (MobileNetV2) on the scalogram images.
5. Evaluate the model performance.

## Output

* Classification reports
* Confusion matrices
* Trained model files
* Scalogram images for each class

## Requirements

* Python 3.x
* Install packages listed in the code using pip (e.g., `pip install neurokit2 tensorflow pywavelets`)

## Author

This project was developed as part of a deep learning capstone project.
