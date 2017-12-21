import numpy as np
from pre_processing import get_data

"""
Data initializing,
First part of the project
"""
X, Y = get_data()
H = 5 # Hidden Units
# Samples or Iterations | Dimensions or Inputs
S, D = X.shape
W = np.random.randn(D, H) # Weights
b = np.random.randn(1, H) # Bias term
K = 4 # Classes
V = np.random.randn(H, K) # Weights Two
c = np.random.randn(1, K) # Bias Two

"""
Classification Rate | Acuracy beetween the predictions 
"""

def classification_rate(Y, P):
    return np.mean(Y == P)

"""
Forward data into the network
Second part of the project

Part One:
    Only Sigmoid
"""

# Defining functions:

def sigmoid(X):
    return 1 / ( 1 + np.exp(-X) )

def forward_sigmoid(W, X, b):
    return sigmoid(X.dot(W) + b)

# Runing functions

z = forward_sigmoid(W, X, b) # First Layer ->

p = forward_sigmoid(V, z, c) # Second Layer ->

final_prediction = np.argmax(p, axis=1)

print(classification_rate(Y, final_prediction))

"""
        Analising the results:
            - The sum overall the dimensions
              of one sample is more than one.

Part Two:
    Softmax
"""

# Defining functions

def softmax(X):
    return np.exp(X) / np.exp(X).sum(axis=1, keepdims=True)

def forward_softmax(W, X, b):
    return softmax(X.dot(W) + b )

# Runing functions

z = forward_softmax(W, X, b) # First Layer ->

p = forward_softmax(V, z, c) # Second Layer ->

final_prediction = np.argmax(p, axis=1)

print(classification_rate(Y, final_prediction))

"""
        Analising the results:
            - The sum overall the dimensions
              of one sample is one.

Part Three:
    Sigmoid and Softmax
"""

# Runing functions

z = forward_sigmoid(W, X, b) # First Layer ->

p = forward_softmax(V, z, c) # Second Layer ->

final_prediction = np.argmax(p, axis=1)

print(classification_rate(Y, final_prediction))

"""
        Conclusion:
        More acurracy with only softmax?
"""
