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

```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader

dataset = pd.read_csv('/customer.csv')
print("Dataset Preview:\n", dataset.head())

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

class PeopleClassifier(nn.Module):
    def __init__(self, input_size, classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size,16)
        self.fc2 = nn.Linear(16,8)
        self.fc3 = nn.Linear(8,classes)

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = PeopleClassifier(X_train.shape[1], len(encoder.classes_))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    for xb,yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out,yb)
        loss.backward()
        optimizer.step()

print("\nTraining Completed")

model.eval()
with torch.no_grad():
    preds = torch.argmax(model(X_test), dim=1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test,preds))

print("\nClassification Report:")
print(classification_report(y_test,preds,target_names=encoder.classes_,zero_division=0))



```



## Dataset Information

<img width="449" height="463" alt="image" src="https://github.com/user-attachments/assets/f6cd4471-dd08-4ed8-a2ab-1165d80570ea" />


## OUTPUT



### Confusion Matrix

<img width="583" height="372" alt="Screenshot 2026-02-05 225926" src="https://github.com/user-attachments/assets/cf748df5-83f9-46e3-aa41-4fcbe98f8ee8" />


### Classification Report
<img width="669" height="273" alt="Screenshot 2026-02-05 225939" src="https://github.com/user-attachments/assets/2ab8f015-4e7d-4c79-a853-853a7598df35" />



### New Sample Data Prediction

<img width="980" height="253" alt="Screenshot 2026-02-05 225859" src="https://github.com/user-attachments/assets/17ba2ff5-9693-47c5-ad18-338671c83bcc" />


## RESULT
Thus, a neural network classification model for the given dataset as been created successfully.
