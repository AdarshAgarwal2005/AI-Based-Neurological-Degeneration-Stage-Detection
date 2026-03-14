# AI-Based Neurological Degeneration Stage Detection

A deep learning project that uses a Convolutional Neural Network (CNN) to classify brain MRI images into different stages of neurological degeneration.
The model analyzes MRI scans and predicts the stage of cognitive decline associated with neurodegenerative conditions such as Alzheimer's disease.

---

## Project Overview

Neurological disorders that affect brain structure and cognitive function often progress through different stages. Early detection of these stages can help in better monitoring and medical intervention.

This project applies deep learning techniques to analyze brain MRI images and automatically classify them into stages of neurological degeneration.

The model is trained on labeled MRI images and learns spatial patterns that indicate brain tissue changes associated with disease progression.

---

## Problem Statement

Manual analysis of MRI scans for detecting neurological degeneration can be time-consuming and requires expert knowledge.

The goal of this project is to develop an AI model that can automatically classify brain MRI images into neurological degeneration stages.

---

## Classes Predicted

The CNN model classifies MRI images into the following four stages:

* **NonDemented** – No signs of neurological degeneration
* **VeryMildDemented** – Early stage cognitive decline
* **MildDemented** – Noticeable memory and thinking impairment
* **ModerateDemented** – Advanced neurological degeneration

These stages represent the progression of neurodegenerative conditions.

---

## Tech Stack

* Python
* PyTorch
* NumPy
* Scikit-learn
* Deep Learning (CNN)

---

## Model Architecture

The model uses a Convolutional Neural Network consisting of:

* Convolutional Layers for feature extraction
* Activation Functions (ReLU)
* Max Pooling layers for dimensionality reduction
* Fully Connected layers for classification

CNNs are highly effective for image-based tasks because they automatically learn spatial features from image data.

---

## Project Structure

```
AI-Neurological-Degeneration-Stage-Detection

data/
    processed/

models/
    trained_model.pth

src/
    train.py
    predict.py
    preprocess.py

test.jpg
requirements.txt
```

---

## Installation

Clone the repository

```
git clone https://github.com/yourusername/AI-Neurological-Degeneration-Stage-Detection.git
```

Move into the project directory

```
cd AI-Neurological-Degeneration-Stage-Detection
```

Create virtual environment

```
python -m venv venv
```

Activate virtual environment

Windows

```
venv\Scripts\activate
```

Install dependencies

```
pip install -r requirements.txt
```

---

## Preprocessing and Training the Model

```
python src/preprocess.py
python src/train.py
```

The trained model will be saved after training.

---


This script loads the trained model and evaluates its accuracy on the test dataset.

---

## Making Predictions

```
python src/predict.py
```

Provide an MRI image and the model will predict the neurological degeneration stage.

---

## Applications

* Medical image analysis
* Early detection of neurodegenerative diseases
* AI-assisted diagnosis systems
* Healthcare research

---

## Future Improvements

* Improve model accuracy using larger datasets
* Use advanced architectures like ResNet or EfficientNet
* Build a web interface for real-time predictions
* Deploy the model using Flask or FastAPI

---

## Author

Adarsh Agrawal
BTech CSE Student
AI / Machine Learning Enthusiast

GitHub: https://github.com/AdarshAgarwal2005
LinkedIn: https://www.linkedin.com/in/adarsh-agarwal-btech-cse/
