# Satellite Image Classification using a Custom CNN

## Overview

This project demonstrates the classification of satellite images using a custom Convolutional Neural Network (CNN) architecture named SpectrumNet. The model is designed to classify images from the EuroSAT dataset into ten distinct land use and land cover classes.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
   - [Data Preprocessing](#data-preprocessing)
   - [Model Building](#model-building)
   - [Model Training](#model-training)
   - [Model Evaluation](#model-evaluation)
   - [Optimizing Model Performance](#optimizing-model-performance)
4. [Results](#results)
5. [Conclusion](#conclusion)
6. [Future Work](#future-work)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Contact](#contact)

## Introduction

Satellite image classification plays a crucial role in applications such as land cover mapping, urban planning, and environmental monitoring. This project leverages deep learning, particularly CNNs, to classify satellite images into ten categories: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, and SeaLake.

## Dataset

The EuroSAT dataset, which comprises 27,000 labeled images of various land use and land cover types, is used for this project. The images are 64x64 pixels and categorized into the following classes:
- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

## Methodology

### Data Preprocessing

1. **Normalization**: Pixel values are scaled to the range [0, 1].
2. **Tensor Conversion**: Images are converted to tensors.
3. **One-Hot Encoding**: Labels are transformed into one-hot encoded vectors.
4. **Dataset Splitting**: The dataset is divided into training, validation, and test sets.
5. **Data Augmentation**: Techniques such as random flips and brightness adjustments are applied to enhance model robustness.

### Model Building

A custom CNN architecture named SpectrumNet was designed, consisting of multiple convolutional and spectral blocks to capture multi-scale features from the input images. The architecture leverages Batch Normalization and ReLU activation to stabilize and accelerate training.

### Model Training

Three versions of the model were trained with different configurations to identify the optimal training parameters. The models were trained using:
- **Optimizer**: Stochastic Gradient Descent (SGD) with momentum
- **Loss Function**: Categorical cross-entropy
- **Class Weights Adjustment**: To handle class imbalance
- **Learning Rate Scheduler**: Step decay function to adjust the learning rate
- **Early Stopping**: To prevent overfitting

### Model Evaluation

The model was evaluated on the test dataset using metrics such as accuracy and confusion matrices to assess its performance and generalization ability.

### Optimizing Model Performance

Three different configurations were tested to determine the best model. The best model achieved a classification accuracy of 96% on the test dataset.

## Results

The best models from each training configuration were compared, and the confusion matrices were analyzed to understand the model's performance across different classes. The training and validation loss and accuracy curves were plotted to visualize the learning process.

## Conclusion

This project demonstrates the effectiveness of CNNs in satellite image classification. The custom SpectrumNet architecture achieved high accuracy, highlighting the importance of meticulous data preprocessing, robust training strategies, and detailed performance evaluation.

## Future Work

Future improvements could include:
- Using higher resolution images to capture more detailed features
- Exploring advanced neural network architectures
- Implementing transfer learning techniques
- Expanding the dataset through advanced augmentation or synthetic data generation

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/AlvaroVasquezAI/Satellite-Image-Classification.git
    cd Satellite-Image-Classification
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Train the model**:
    ```bash
    python train.py
    ```

2. **Evaluate the model**:
    ```bash
    python evaluate.py
    ```

3. **Make predictions**:
    ```bash
    python predict.py --image_path path/to/image.jpg
    ```

## Contact

For any questions or inquiries, feel free to contact me:

- Email: [agarciav2102@alumno.ipn.mx](mailto:agarciav2102@alumno.ipn.mx)
- GitHub: [Álvaro Vásquez AI](https://github.com/AlvaroVasquezAI)
- LinkedIn: [Álvaro García Vásquez](https://www.linkedin.com/in/%C3%A1lvaro-garc%C3%ADa-v%C3%A1squez-8a2a001bb/)
- Instagram: [alvarovasquez.ai](https://www.instagram.com/alvarovasquez.ai)
- Twitter: [alvarovasquezai](https://twitter.com/alvarovasquezai)
- YouTube: [Álvaro Vásquez AI](https://www.youtube.com/channel/UCd8GEklq1EbrxGQYK0CXDTA)

Feel free to contribute to this project by opening issues and submitting pull requests.

---

This project demonstrates the potential of deep learning in remote sensing applications and provides a foundation for future advancements in satellite image classification.
