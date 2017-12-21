"""
Backpropagation functions to Train the network
"""
import numpy as np
import prediction

def classification_rate(Y, P):
    return np.mean(Y != P)

def main():

    