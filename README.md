**Unveiling the Anomalies: Using Autoencoders for Detecting Structural Changes in Huntington's Disease from MRI Images**

**Table of Contents**
1. Introduction
2. Project Objective
3. Methodology
4. Technologies Used
5. Model Architecture
6. Results

**Introduction**
Huntington's Disease (HD) is a genetic neurodegenerative disorder that causes the progressive breakdown of nerve cells in the brain, leading to cognitive and motor function impairments. The ability to detect early structural changes in the brain is critical for early diagnosis and intervention.
This project uses Autoencoders to identify structural anomalies from MRI scans, which are key indicators of Huntington’s Disease progression. By detecting these changes, the model aims to assist in the early diagnosis and monitoring of the disease.

**Project Objective**
The objective of this project is to develop a machine learning model that can automatically detect and highlight structural changes in MRI images of the brain, which are associated with the progression of Huntington's Disease. The project focuses on: 
1. Using MRI imaging as the primary data source.
2. Employing autoencoders for unsupervised anomaly detection.
3. Providing insights into the regions of the brain affected by HD.

**Methodology**
The project follows a pipeline that includes:
1. Preprocessing: MRI scans are normalized, resized, and augmented to enhance model performance.
2. Autoencoder Training: An unsupervised learning model is trained on healthy brain images, learning to compress and reconstruct them.
3. Anomaly Detection: The model evaluates test scans (both healthy and affected), flagging anomalies by comparing reconstructed outputs with the original input.
4. Post-Processing: Visualization tools are used to highlight regions with significant deviations, indicating areas of potential structural changes.

**Technologies Used**
1. Python
2. TensorFlow/Keras for building the Autoencoder model
3. OpenCV and Numpy for image preprocessing
4. Scikit-learn for data splitting and evaluation metrics
5. Matplotlib/Seaborn for result visualization

**Model Architecture**
The autoencoder model consists of:
1. Encoder: A series of convolutional layers that compress the input image into a lower-dimensional latent space.
2. Decoder: A symmetric set of convolutional layers that reconstruct the input from the latent space.
The reconstruction error (difference between the original image and its reconstruction) is used to identify anomalies in the test MRI scans.

**Results**
The model successfully detects anomalies in the test MRI images. These anomalies correspond to regions in the brain affected by Huntington’s Disease. The model demonstrates a good ability to distinguish between healthy brain images and those with structural changes caused by HD.

Evaluation metrics used include:
1. Reconstruction Loss: Measures how well the model can recreate healthy MRI images.
2. Mean Squared Error (MSE): Between the input and reconstructed output.
3. Visualization: Anomalies are highlighted by differences between the input image and the reconstruction.
