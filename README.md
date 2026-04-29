# Lung Sound Disease Classification

CNN-based biomedical audio classifier achieving 80-92% accuracy across 8 respiratory disease classes using MFCC feature extraction.

## Overview

End-to-end pipeline for classifying respiratory diseases from lung sound recordings. Raw audio is preprocessed using MFCC (Mel-Frequency Cepstral Coefficients) and fed into a CNN trained on multi-class labelled data.

Built during Vocational Training at IIIT Naya Raipur, Biomedical AI Lab, Summer 2025.

## Results

| Metric | Value |
|---|---|
| Best Accuracy | 92% |
| Disease Classes | 8 |
| Validation | 5-Fold Cross Validation |
| Dataset Size | 6000+ audio samples |
| Framework | TensorFlow/Keras and PyTorch |

## Tech Stack

- TensorFlow / Keras: Model training
- PyTorch: Alternate implementation
- Librosa: Audio processing and MFCC extraction
- scikit-learn: Cross-validation and metrics
- Matplotlib / Seaborn: Confusion matrix and accuracy plots

## Pipeline

Raw Audio -> Preprocessing -> MFCC Feature Extraction (40 coefficients) -> 4-Layer CNN -> Multi-class Output (8 classes) -> Evaluation

## How to Run

git clone https://github.com/Ayanmishra11/respiratory_disease_classification
cd respiratory_disease_classification
pip install -r requirements.txt
jupyter notebook

## Author

Ayan Mishra, B.Tech CSE (AI), BIT Durg
Email: ayanmishra9820@gmail.com
LinkedIn: https://linkedin.com/in/ayan-mishra-971bab299
