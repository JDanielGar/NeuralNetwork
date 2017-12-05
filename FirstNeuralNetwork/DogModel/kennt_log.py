# coding: utf-8
import pandas as pd
import numpy as np

datos = pd.read_excel('./datos_kenny.xls').as_matrix()

datos = datos[:, 1:]

X = datos[:, :3]
Y = datos[:, 3:]

# Cambiar tipo de objeto

X = X.astype(float)
Y = Y.astype(float)

# Normalización

X[:, 2:3] = (X[:, 2:3] - X[:, 2:3].mean()) / X[:, 2:3].std()


# Datos de Entrenamiento y de Test

X_train, Y_train = X[:7, :], Y[:7, :]
X_test, Y_test = X[7:, :], Y[7:, :]

# Pesos sinapticos inicializados aleatoriamente

S, D = X.shape
W = np.random.rand(D, 1)

# Hiperparametros

learning_rate = 0.001
b = 0

# Forward functions 

def sigmoid(X):
    return 1/(1 + np.exp(-X))

def forward(W, X, b):
    return sigmoid(X.dot(W) + b)

# Cross Entropy cost function

def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY) + (1 - T)*np.log(1 - pY))

# Clasification rate

def classification_rate(Y, P):
    return np.mean(Y == P)

train_costs = [] 
test_costs = []

# Training Part

for epoch in range(1000):
    pY_train = forward(W, X_train, b)
    pY_test = forward(W, X_test, b)
    cost_train = cross_entropy(Y_train, pY_train)
    cost_test = cross_entropy(Y_test, pY_test)

    # COST

    train_costs.append(cost_train)
    test_costs.append(cost_test)

    # GRADIENT DESCENT... THE FUNNIEST PART

    W -= learning_rate * X_train.T.dot(pY_train - Y_train)
    b -= learning_rate * (pY_train - Y_train).sum()

    print(epoch, cost_train, cost_test)
        
print("Final train Classification rate", classification_rate(Y_train, np.round(pY_train)))
print("Final test classification rate", classification_rate(Y_test, np.round(pY_test)))