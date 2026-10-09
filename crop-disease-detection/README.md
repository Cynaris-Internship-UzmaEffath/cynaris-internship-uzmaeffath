# Crop Disease Detection

This project is about detecting crop diseases from leaf images using a CNN model.

The user can upload a leaf image, and the system will predict the disease, show the confidence score, and give a basic treatment suggestion.

## Technologies Used

- Python
- PyTorch
- CNN
- ResNet18
- FastAPI
- Pandas
- NumPy
- Pillow
- Scikit-learn

## How It Works

Leaf photo

-> Image preprocessing

-> Data augmentation

-> ResNet18 model

-> Disease prediction

-> Confidence score

-> Treatment suggestion

-> FastAPI response

## Dataset

The main dataset for this project is the PlantVillage dataset.

The project also provides these Agrimind files:

- agrimind_crop_disease_data.csv
- agrimind_market_prices.csv
- agrimind_test_cases.json

The Agrimind files currently return a 404 error, so they are pending until the links are fixed.

## Model

I am using a pretrained ResNet18 model with transfer learning to classify crop diseases.

## Evaluation

The model will be checked using:

- Accuracy
- Precision
- Recall
- F1-score

The target is to achieve at least 88% accuracy on the test set.

## Project Structure

crop-disease-detection/
|
|-- api/
| |-- main.py
|
|-- data/
| |-- raw/
| |-- processed/
|
|-- models/
|
|-- notebooks/
|
|-- src/
| |-- data_pipeline.py
| |-- model.py
| |-- train.py
| |-- evaluate.py
| |-- treatment_rules.py
|
|-- README.md
|-- requirements.txt
|-- .gitignore

## Setup

Create the virtual environment:
python -m venv .venv
Activate it:
.\.venv\Scripts\Activate.ps1
Install the required packages:
pip install -r requirements.txt

## Research

I reviewed three resources related to crop disease detection.

1. "Using Deep Learning for Image-Based Plant Disease Detection (2016)"

   This paper used CNNs with the PlantVillage dataset for plant disease detection. It helped me understand how deep learning can be used for this type of image classification.

2. "Sustainable AI for Plant Disease Classification Using ResNet18 (2025)"

   This study used ResNet18 and transfer learning for plant disease classification. It helped me decide to use ResNet18 for this project.

3. "Official PlantVillage GitHub Repository"

   This repository provides the PlantVillage dataset and information about the dataset. It helped me understand where the main dataset comes from.

### What I Learned

From the research, I decided to use ResNet18 with transfer learning. I will also use image preprocessing and augmentation before training the model.

## Current Progress

- [x] Project setup
- [x] Basic architecture
- [x] Model selected
- [x] Evaluation metrics selected
- [x] Research
- [x] Data pipeline skeleton
- [ ] Dataset preparation
- [ ] Model training
- [ ] Model evaluation
- [ ] Treatment rules
- [ ] FastAPI API
- [ ] Local testing

## Goal

The goal is to build a simple system that can identify crop diseases from leaf images and provide the disease name, confidence score, and treatment suggestion.
