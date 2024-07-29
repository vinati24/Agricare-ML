# Agricultural Advisory System

This repository contains three modules: **Crop Recommendation**, **Fertilizer Recommendation**, and **Disease Detection**. These modules utilize machine learning models to assist farmers and agricultural experts in making informed decisions.

## Modules

### 1. Crop Recommendation
The Crop Recommendation module uses machine learning models to analyze historical data on weather, soil conditions, crop yields, and pH levels. Based on this analysis, it recommends the most suitable crop for a specific location and season. This helps in optimizing crop selection and improving overall yield.

### 2. Fertilizer Recommendation
The Fertilizer Recommendation module analyzes various features such as temperature, soil type, and crop type to predict the best-suited fertilizer for a particular crop. By providing precise fertilizer recommendations, this module aims to maximize crop yield and ensure efficient use of resources.

### 3. Disease Detection
The Disease Detection module helps in identifying plant diseases. Users can upload images of diseased plants, leaves, or flowers, and the system will analyze these images to predict the disease. This module leverages the RESNET9 model, a deep learning architecture, to provide accurate disease identification.


### Prerequisites
- Python 3.x
- Required libraries: `numpy`, `pandas`, `scikit-learn`, `tensorflow`, `keras`, `opencv-python`
