<br/>
<p align="center">
  <h1 align="center">Satellite Image Classification</h1>

  <p align="center">
    Using a CNN
    <br />
  </p>
</p>
<!-- TECHNOLOGY BADGES -->
<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Matplotlib-3776AB?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Tkinter-white?style=for-the-badge&logo=python&logoColor=blue" alt="Tkinter">
</p>

This project demonstrates the classification of satellite images using a custom Convolutional Neural Network (CNN). The model is designed to classify images from the EuroSAT dataset into ten distinct classes.

<p align="center"><img src = "Resources/Satellite Image Classification.png" width="800"/></p>

## Table of contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
   - [Data preprocessing](#data-preprocessing)
   - [Model building](#model-building)
   - [Model training](#model-training)
   - [Model evaluation](#model-evaluation)
   - [Optimizing model performance](#optimizing-model-performance)
4. [Results](#results)
5. [Conclusion](#conclusion)
6. [Future work](#future-work)
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

<p align="center">
   <img src = "Resources/Classes/AnnualCrop.jpg"/>
   <img src = "Resources/Classes/Forest.jpg"/>
   <img src = "Resources/Classes/HerbaceousVegetation.jpg"/>
   <img src = "Resources/Classes/Highway.jpg"/>
   <img src = "Resources/Classes/Industrial.jpg"/>
   <img src = "Resources/Classes/Pasture.jpg"/>
   <img src = "Resources/Classes/PermanentCrop.jpg"/>
   <img src = "Resources/Classes/Residential.jpg"/>
   <img src = "Resources/Classes/River.jpg"/>
   <img src = "Resources/Classes/SeaLake.jpg"/>
</p>

## Methodology

### Data Preprocessing

1. **Normalization**: Pixel values are scaled to the range [0, 1].
2. **Tensor conversion**: Images are converted to tensors.
3. **One-Hot encoding**: Labels are transformed into one-hot encoded vectors.
4. **Dataset splitting**: The dataset is divided into training, validation, and test sets.
5. **Data augmentation**: Techniques such as random flips and brightness adjustments are applied to enhance model robustness.

### Model building

A custom CNN architecture named SpectrumNet was designed, consisting of multiple convolutional and spectral blocks to capture multi-scale features from the input images. The architecture leverages Batch Normalization and ReLU activation to stabilize and accelerate training.

<p align="center"><img src = "Resources/architecture_cnn.png" width="100"/></p>

### Model training

Three versions of the model were trained with different configurations to identify the optimal training parameters. The models were trained using:
- **Optimizer**: Stochastic Gradient Descent (SGD) with momentum
- **Loss function**: Categorical cross-entropy
- **Class weights adjustment**: To handle class imbalance
- **Learning rate scheduler**: Step decay function to adjust the learning rate
- **Early stopping**: To prevent overfitting

### Model evaluation

The model was evaluated on the test dataset using metrics such as accuracy and confusion matrices to assess its performance and generalization ability.

### Optimizing model performance

Three different configurations were tested to determine the best model. The best model achieved a classification accuracy of 96% on the test dataset.

## Results

### Predictions
<p align="center">
   <img src = "Resources/labels.png" width="700"/>
</p>

<p align="center">
  <img src="Resources/Results/Result1.png" width="700"/>
</p>
<p align="center">
  <img src="Resources/Results/Result2.png" width="700"/>
</p>
<p align="center">
  <img src="Resources/Results/Result3.png" width="700"/>
</p>


## Conclusion

This project demonstrates the effectiveness of CNNs in satellite image classification. The custom SpectrumNet architecture achieved high accuracy, highlighting the importance of meticulous data preprocessing, robust training strategies, and detailed performance evaluation.

## Future work

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
### Use the GUI for classification

To use the graphical interface for classifying satellite images:

1. Run the GUI application:
    ```bash
    python classifierApp.py
    ```

2. **Select a Model**: Click on the "Select Model" button and choose the trained model file (`.keras`).

3. **Select an Image**: Once a model is loaded, click on the "Select Image" button and choose the satellite image you want to classify.

4. **View Results**: The application will display the original image and the colorized classification map. You can select a different image to classify or load a new model to use for classification.

<p align="center">
   <img src = "Resources/InterfazResult.png" width="800"/>
</p>

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
