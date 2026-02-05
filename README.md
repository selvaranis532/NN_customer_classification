# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="787" height="787" alt="image" src="https://github.com/user-attachments/assets/13a1ab90-3062-45c3-8ed3-0ac72833deae" />

DESIGN STEPS
3 STEP 1: Import necessary libraries and load the dataset.

STEP 2:
Encode categorical variables and normalize numerical features.

STEP 3:
Split the dataset into training and testing subsets.

STEP 4:
Design a multi-layer neural network with appropriate activation functions.

STEP 5:
Train the model using an optimizer and loss function.

STEP 6:
Evaluate the model and generate a confusion matrix.

STEP 7:
Use the trained model to classify new data samples.

STEP 8:
Display the confusion matrix, classification report, and predictions.


## PROGRAM

### Name: SELVARANI S
### Register Number:212224040301

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        #Include your code here



    def forward(self, x):
        #Include your code here
        

```
```python
# Initialize the Model, Loss Function, and Optimizer


```
```python
def train_model(model, train_loader, criterion, optimizer, epochs):
    #Include your code here
```



## Dataset Information

Include screenshot of the dataset

## OUTPUT



### Confusion Matrix

Include confusion matrix here

### Classification Report

Include Classification Report here


### New Sample Data Prediction

Include your sample input and output here

## RESULT
Thus, a neural network classification model for the given dataset as been created successfully.
