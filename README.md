# Var-CNN Project

## Overview

The AI-VAR project is a deep learning model for classifying clean tackles and fouls in football images. It utilizes a c custom convolutional neural network (VarCNN) to achieve this classification.

## Table of Contents

- [Getting Started](#getting-started)
- [Implementation](#implementation)
- [Model Structure and Explaination](#project-structure)
- [Future Work](#future)

## Getting Started

To get started with the Var-CNN project, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/Var-CNN.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the project.

## Implementation

Provide information on how to use your project, including:
- Data Collection: 
    + The image data is collected from this repository: https://github.com/aamir09/VarCnn
    + You can download and extract it in a folder named "model_data" at the root directory
    + Data structure: 
        - 2 folders: test and train 
        - In each folder: clean tackles and fouls

- Training the model
    
    Soccer stands out as the most widely followed sport globally. Over the past century, the sport has undergone significant developments, including advancements in its technological aspects. Among these innovations, the Virtual Assistant Referee (VAR) has emerged as a game-changer, exerting a considerable influence on the sport. The VAR's role is both straightforward and intricate – it steps in during the game when referees make erroneous decisions or find it challenging to make one. A particular situation arises when determining whether a sliding tackle within the penalty box should be considered a clean tackle or lead to a penalty for the opposing team. While some other times, it is important to determine whether a sliding tackle can be enough for a red card. While technology allows for reviewing the moment of the tackle repeatedly, the final decisions still rest with humans, introducing the potential for bias. In this project, I propose a Convolutional Neural Network (CNN)-based approach to foul detection, grounded in the concept of analyzing the initial point of contact.


- Model Structure: 
    ![Model Structure](https://github.com/duongdang1/AI_VAR/blob/72ef6641859f296a76112787b87b1c39d3c717be/VarCNN.drawio.png)

- Result and Analysis: 
    ![Train Result](https://github.com/duongdang1/AI_VAR/blob/ec3f15f24572c923c89fb8594508d90bba24d9a8/85%25.png)

- Analysis
    1. Improvement over training
        + The model has shown significant improvement throughout the training process. The initial accuracy was around 49.84%, and it increased to 85.67% by the last epoch.
        + The corresponding loss has also reduced from 0.6961 to 0.3313, indicating that the model is learning and making better predictions.
    
    2. Test Accuracy:
        The test accuracy also increased from 49.38% in the initial epoch to 71.37% in the last epoch. This suggests that the model is generalizing well to unseen data.
    
    3. Early Stopping:
        The early stopping mechanism was triggered after 10 consecutive epochs without improvement in the test loss. This indicates that the model might have reached a point where further training does not lead to better generalization.

    4. Compare to previous models (https://github.com/aamir09/VarCnn/tree/main)
        Previous Project:
        - Training Accuracy: 76.6%
        - Validation Accuracy: 78%
        Current Project (Last Epoch):
        - Training Accuracy: 85.67%
        - Test Accuracy: 71.37%    

        Several factors could be contributed to why the train accuracy improved but the test/validation accuracy decreased: 
        1. Overfitting: the model might be overfitting to the training data
        2. Data Mismatch: 
        3. Hyperparameter Tuning

## Model Structure and Explaination

1. Input Layer: The model takes an input image with three channels (RGB).

2. Convolutional Layer 1 (`self.conv1`): 
    - Applies the first convolution with 64 filters, a kernel size of 5, and padding of 2.
    - Followed by ReLU activation to introduce non-linearity.
    - MaxPooling layer with a kernel size of 2 and a stride of 2 reduces spatial dimensions.

3. Convolutional Blocks (`self.conv_blocks`):
    - Consists of four sets of Conv2d-ReLU-MaxPool layers.
    - Each set has a Conv2d layer with 64 filters, followed by ReLU activation, and a MaxPool layer.
    - These blocks extract hierarchical features from the input image.

4. Dilation Blocks (`self.dilation_blocks`):
    - Consists of three sets of Conv2d-ReLU-MaxPool layers with dilated convolutions.
    - Dilation is applied to capture larger spatial context without increasing the number of parameters.

5. Dense Layers (`self.dense_layers`):
    - AdaptiveAvgPool2d layer reduces the spatial dimensions to 1x1.
    - Flatten layer reshapes the tensor for fully connected layers.
    - Three fully connected layers with 400, 512, and 400 neurons, respectively.
    - ReLU activations introduce non-linearity.
    - Dropout layers with a dropout rate of 0.5 to prevent overfitting.

6. Output Layer (`self.sigmoid`):
    - Applies a sigmoid activation function to obtain binary classification probabilities.
    - The model aims to predict whether the input image belongs to a certain class (binary classification).


## Future Work
1. Data augmentation: 
    - Experiment more techniques to enhance model generalization
2. Hyperparameter Tuning: 
    - Explore different learning rates, weight, dropout rates 
3. Explainability and Interpretability: 
    - Explore methods such as Grad-CAM to visualize which part of the image contribute most to the model's precision