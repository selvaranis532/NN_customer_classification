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
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4) 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  
        return x
```
```
# Initialize the Model, Loss Function, and Optimizer
model = PeopleClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(), lr=0.001)

```
```
def train_model(model, train_loader, criterion, optimizer, epochs):
  for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```



## Dataset Information

<img width="1492" height="290" alt="image" src="https://github.com/user-attachments/assets/94fe2b85-add9-4ec2-b36d-52c5a1833254" />



## OUTPUT



### Confusion Matrix

<img width="772" height="654" alt="image" src="https://github.com/user-attachments/assets/c59ed3b2-9df2-4a61-85a6-78e243f9037a" />



### Classification Report
<img width="663" height="438" alt="image" src="https://github.com/user-attachments/assets/d76b0636-65f0-4604-b2eb-97d8944729ab" />




### New Sample Data Prediction

<img width="1197" height="323" alt="image" src="https://github.com/user-attachments/assets/02b336ec-3edb-42c9-98ee-440d92bbf078" />


## RESULT
Thus, a neural network classification model for the given dataset as been created successfully.
