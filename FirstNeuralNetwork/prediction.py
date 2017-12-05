"""

-----
Daniel Garcia
03/12/17
-----

FeedForward:
Training set is a combination of inputs X and targets Y
N will be the number of samples and D the number of features

X = Inputs
Y = Targets
D = Features
N = Samples

Now we gonna try out two experiments:
    - Binary Clasification
    - Two Clasification

Structure of neural net:

        X               T   HS      S       T = Has a tecnical degree? (1=Yes, 2=Not)
    I                   0   3.5     1       HS = Hours spending.
        X   O           1   2       1       S = Susess course?
    I                   1   0.5     0
        X

I = Inputs
X =  Hidden Layer
O = Output

"""

import numpy as np

# Binary Clasification

X = np.array([[0, 3.5], [1, 2], [1, 0.5]])
Y = np.array([[1, 1, 0]]) # Only used in training,   not prediction.

# Weights in the first hidden layer

W = np.array([[0.5, 0.1, -0.3], [0.7, -0.3, 0.2]])  

# Bias Term in the first hidden layer

b = np.array([[0.4, 0.1, 0]])

# Weights of the hidden to the output layer

V = np.array([[0.8, 0.1, -0.1]])

# Bias term in the hidden to the output layer

c = 0.2

# Forward PART 

def forward(W, X, b):
    return np.tanh(X.dot(W) + b)

first_layer = forward(W, X[0], b)
output = forward(V.T, first_layer, c)

def sigmoid(x):
    return 1/(1+np.exp(-X))

def forward_with_layers(W, X, b, V, c):
    return sigmoid(V.dot(np.tanh(X.dot(W)+b))+c)

forward_with_layers(W, X, b, V, c)