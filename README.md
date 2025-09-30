# EX-1: Developing a Neural Network Regression Model

## AIM:

To develop a neural network regression model for the given dataset.

## THEORY:

Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model:

<img width="822" height="674" alt="image" src="https://github.com/user-attachments/assets/d1936d43-7ba1-4262-88da-ef709a66cb9f" />


## DESIGN STEPS:

**STEP 1: Generate Dataset**
Create input values from 1 to 50 and add random noise to introduce variations in output values .

**STEP 2: Initialize the Neural Network Model**
Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

**STEP 3: Define Loss Function and Optimizer**
Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

**STEP 4: Train the Model**
Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

**STEP 5: Plot the Loss Curve**
Track the loss function values across epochs to visualize convergence.

**STEP 6: Visualize the Best-Fit Line**
Plot the original dataset along with the learned linear model.

**STEP 7: Make Predictions**
Use the trained model to predict for a new input value .

## PROGRAM:

**Name**: Hariharan S

**Register Number:** 2305001009

```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df=df.astype({'INPUT':'float'})
df=df.astype({'OUTPUT':'float'})
df.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = df[['INPUT']].values
y = df[['OUTPUT']].values
X

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)

model=Sequential([
    #Hidden ReLU Layers
    Dense(units=5,activation='relu',input_shape=[1]),
    Dense(units=3,activation='relu'),
    #Linear Output Layer
    Dense(units=1)
])

model.compile(optimizer='rmsprop',loss='mse')
model.fit(X_train1,y_train,epochs=3000)

loss= pd.DataFrame(model.history.history)
loss.plot()

X_test1 =Scaler.transform(X_test)
model.evaluate(X_test1,y_test)

X_n1=[[4]]
X_n1_1=Scaler.transform(X_n1)
model.predict(X_n1_1)
```

Dataset Information

<img width="538" height="453" alt="image" src="https://github.com/user-attachments/assets/d823d195-c5e1-4b88-ab67-97234288c838" />


# OUTPUT:

Training Loss Vs Iteration Plot:

<img width="560" height="413" alt="image" src="https://github.com/user-attachments/assets/2cc61e88-b67e-4b34-bf6e-d8facc3f7af7" />

Epoch Training:
<img width="1032" height="337" alt="image" src="https://github.com/user-attachments/assets/71af18e0-4704-484b-add6-44124c1c6723" />

Test Data Root Mean Squared Error:

<img width="768" height="66" alt="image" src="https://github.com/user-attachments/assets/545d0a64-a7c7-4cdb-b86a-84e515151055" />

New Sample Data Prediction:

<img width="641" height="65" alt="image" src="https://github.com/user-attachments/assets/f4719225-390e-41a0-ae64-07ee877e3dfd" />

# RESULT:

Thus, a neural network regression model was successfully developed and trained using PyTorch.
