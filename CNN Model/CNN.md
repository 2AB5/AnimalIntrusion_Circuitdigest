Overview

This repository contains a Convolutional Neural Network (CNN) built using PyTorch to classify images as either Elephant or Non-Elephant. The model was designed to be lightweight and efficient for deployment on edge devices like the MAX78000FTHR, making it ideal for real-time animal recognition in resource-constrained environments.
Model Architecture

The CNN model consists of the following layers:

    Input Layer:
        Input Size: 64x64 RGB images (3 channels)

    Convolutional Layer 1:
        Number of Filters: 16
        Filter Size: 3x3
        Activation: ReLU
        Output Size: 64x64x16 (16 feature maps)

    Max-Pooling Layer 1:
        Pooling Size: 2x2
        Stride: 2
        Output Size: 32x32x16 (Feature map size is halved)

    Convolutional Layer 2:
        Number of Filters: 32
        Filter Size: 3x3
        Activation: ReLU
        Output Size: 32x32x32 (32 feature maps)

    Max-Pooling Layer 2:
        Pooling Size: 2x2
        Stride: 2
        Output Size: 16x16x32 (Feature map size is halved)

    Fully Connected Layer:
        Input Size: Flattened 8192 (32 channels * 16x16 spatial dimension)
        Output Size: 128 units
        Activation: ReLU

    Output Layer:
        Number of Units: 2 (for Elephant and Non-Elephant classification)
        Activation: Softmax

Training Details

    Loss Function: CrossEntropyLoss
    Optimizer: Adam with learning rate 0.001
    Dataset: A custom dataset containing labeled images of elephants and non-elephants, structured into training, validation, and test sets.

Datesets used 
    [Animals10 - Alessiocorrado99](https://www.kaggle.com/datasets/alessiocorrado99/animals10) \\
    [Sri-lankan-wild-elephant-dataset - gunarakulangr](https://www.kaggle.com/datasets/gunarakulangr/sri-lankan-wild-elephant-dataset) \\
    [Github.com/Thisun](https://github.com/ThisunT/Elephant-Identification-System/tree/master/back-end/retraining/dataset)
