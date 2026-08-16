# 🔬 AI-Based Myopia Classification System

An AI-based medical image classification system designed to classify eye images into **Normal**, **Pathological Myopia**, and **High Myopia** using deep learning.

The project involves extensive experimentation with multiple deep learning architectures, activation functions, hyperparameter configurations, and model evaluation techniques. The best-performing model is integrated into a **Streamlit-based web application** for real-time image classification.

> **Note:** This project is intended as a preliminary classification/research prototype and is not a replacement for professional medical diagnosis.

---

## 🚀 Project Overview

This project focuses on **deep learning-based medical image classification** for myopia severity assessment.

Multiple deep learning architectures were trained and evaluated using the eye image dataset. Different model configurations, activation functions, and hyperparameter values were experimented with to understand their impact on model performance.

The experiments were performed across different configurations, including **temperature values ranging from 0.2 to 0.6**, along with different activation-function configurations.

After comparing the experimental results, the best-performing configuration was selected and integrated into the Streamlit application.

The application provides a complete workflow for image upload, prediction, result storage, and patient history visualization.

---

## ✨ Key Features

- Eye image classification:
  - **Normal**
  - **Pathological Myopia**
  - **High Myopia**
- Multiple deep learning architecture experiments
- Model training and performance comparison
- Image preprocessing and normalization
- Activation-function experimentation
- Hyperparameter experimentation
- Temperature-based configuration experiments (**0.2 – 0.6**)
- Model evaluation using training and validation performance
- Prediction confidence analysis
- Streamlit-based web application
- Doctor/Admin and Patient interaction workflow
- Patient history storage using SQLite
- Modular and extensible project structure

---

## 🧠 Deep Learning Models Evaluated

Multiple architectures were trained and evaluated to compare their classification performance.

### Convolutional & Deep Learning Architectures

1. **CNN** — Convolutional Neural Network
2. **RNN** — Recurrent Neural Network
3. **GoogLeNet** — Inception architecture
4. **DenseNet** — Densely Connected Convolutional Network
5. **ResNet-50** — Residual Network
6. **ResNet-101** — Residual Network
7. **ResNet-152** — Residual Network
8. **VGG16** — Visual Geometry Group architecture
9. **VGG19** — Visual Geometry Group architecture
10. **MobileNet** — Lightweight convolutional architecture
11. **InceptionV3** — Advanced Inception architecture

Each architecture was trained and evaluated to determine its suitability for the given eye image classification task.

---

## 🧪 Model Experimentation

The project involved experimentation beyond simply training a single model.

Different configurations were tested to understand their impact on model performance.

### Activation Functions

Different activation-function configurations were evaluated during model experimentation, including commonly used functions such as:

- **ReLU**
- **Sigmoid**
- **Tanh**
- **Softmax**

The activation functions were selected according to their role within the network, such as hidden layers and classification output layers.

### Temperature Experimentation

Different temperature values were evaluated as part of the model configuration and prediction experiments.

Tested range:

```text
0.2
0.3
0.4
0.5
0.6
