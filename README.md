# Var-CNN Project

## Overview

The AI-VAR project is a deep learning model for classifying clean tackles and fouls in football images. It utilizes a c custom convolutional neural network (VarCNN) to achieve this classification.

## Table of Contents

- [Getting Started](#getting-started)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

To get started with the Var-CNN project, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/Var-CNN.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the project.

## Usage

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
    ![Model Structure]](https://github.com/duongdang1/AI_VAR/blob/72ef6641859f296a76112787b87b1c39d3c717be/VarCNN.drawio.png)

- Any command-line arguments or configurations

## Project Structure

Describe the structure of your project. Highlight important directories and files.

