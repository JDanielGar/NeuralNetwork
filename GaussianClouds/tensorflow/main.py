import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

Nclass = 500
D = 2
M = 3
K = 3

X1 = np.random.randn(Nclass, D) + np.array([0, -2])
X2 = np.random.randn(Nclass, D) + np.array([2, 2])
X3 = np.random.randn(Nclass, D) + np.array([-2, 2])

X = np.vstack([X1, X2, X3]).astype(np.float32)
Y = np.array([0] * Nclass + [1] * Nclass + [2] * Nclass) 
N = len(Y)

T = np.zeros((N, K))
for i in range(N):
    T[i, Y[i]] = 1 