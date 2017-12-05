# coding: utf-8
"""
Daniel García
03/12/17
"""
# Basic Unit of Neural Network

# [happends?, importance]

obesity = [True, 1]
smokes = [True, 0.5]
exercise = [False, -0.8]

# Predict the lineals

def linear(w, x, b=0): 
    return w*x+b

obesity_prediction = linear(obesity[1], obesity[0])
smokes_prediction = linear(smokes[1], smokes[0])
exercise_prediction = linear(exercise[1], exercise[0])

# Sum the results

result = obesity_prediction + smokes_prediction + exercise_prediction

# Apply the Sigmoid function

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

result = sigmoid(result)

print('The probability of have a pulmonar disease is:', result)

# Forward is the common name in ML of neural network going forward, couse neural network can go backwards

def forward(w, x, b=0):
    return sigmoid(w.dot(x) + b)

