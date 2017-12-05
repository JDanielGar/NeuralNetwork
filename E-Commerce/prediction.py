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
Forward data into the network
Second part of the project

Part One:
    Only Sigmoid
"""

# Defining functions:

def sigmoid(X):
    return 1 / ( 1 + np.exp(-X) )

def forward(W, X, b):
    return sigmoid(X.dot(W) + b)

# Runing functions

z = forward(W, X, b) # First Layer ->

p = forward(V, z, c) # Second Layer ->

"""
        Analising the results:
            - The sum overall the dimensions
              of one sample is more than one.
"""