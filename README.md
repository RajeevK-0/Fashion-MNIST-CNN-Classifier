# Fashion MNIST Image Classification using CNN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📌 Project Overview
This project implements a **Convolutional Neural Network (CNN)** to classify images from the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist). The model identifies 10 different categories of clothing (e.g., T-shirts, Sneakers, Bags) using a custom deep learning architecture built with **TensorFlow** and **Keras**.

The goal of this project is to demonstrate:
* Data preprocessing pipelines (normalization, reshaping).
* Building a Functional API model in Keras.
* Preventing overfitting using Regularization techniques (Dropout & Batch Normalization).
* Model evaluation and serialization.

## 📊 The Dataset
* **Source:** Fashion MNIST (Keras datasets)
* **Training Set:** 60,000 images
* **Test Set:** 10,000 images
* **Image Size:** 28x28 grayscale pixels
* **Classes:** 10 (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

## 🧠 Model Architecture
The model utilizes a Sequential-style architecture optimized for spatial feature extraction:

1.  **Input Layer:** (28, 28, 1)
2.  **Conv Block 1:** Conv2D (32 filters, 3x3) + ReLU + MaxPool2D
3.  **Conv Block 2:** Conv2D (32 filters, 3x3) + ReLU + MaxPool2D
4.  **Regularization:** BatchNormalization + Dropout (0.1)
5.  **Dense Layers:** Flatten -> Dense(512) -> Dense(256) -> Dense(128)
6.  **Output:** Dense(10) with Softmax activation

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Data Manipulation:** NumPy
* **Visualization:** Matplotlib

## 🚀 How to Run

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/fashion-mnist-cnn.git](https://github.com/yourusername/fashion-mnist-cnn.git)
    cd fashion-mnist-cnn
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train the model**
    ```bash
    python fashionmnist.py
    ```
    *This will train the model for 10 epochs, display a visualization of the dataset, and save the final model as `fashion_mnist_model.keras`.*

## 📈 Results
* **Optimizer:** Adam
* **Loss Function:** Sparse Categorical Crossentropy
* **Test Accuracy:** ~91% (Varies slightly per run)
