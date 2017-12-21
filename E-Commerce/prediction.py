"""
Forward predictions to do backpropagation
"""

import numpy as np
import matplotlib.pyplot as plt

def forward(X, W1, b1, W2, b2):
    Z = 1 / 1 - np.exp(-X.dot(W1) - b1)
    A = Z.dot(W2) + b2
    return softmax(A)

def softmax(X):
    exp = np.exp(X)
    return exp / exp.sum(axis=1, keepdims=True)
