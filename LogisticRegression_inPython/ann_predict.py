import numpy as np
from preprocessing_data import get_data

X, Y = get_data()

M = 5 # hidden unots
D = X.shape[1] #num of categories
K = len(set(Y)) #num of different numbers in Y

W1 = np.random.randn(D,M)
b1  = np.zeros(M)
W2 = np.random.randn(M,K)
b2 = np.zeros(K)

def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis = 1, keepdims=True)

def forward_action(X, W1, b1, W2, b2):
    Z  = np.tanh(X.dot(W1) + b1) # hidden layer values via tanh activation function
    return softmax(Z.dot(W2)+b2)

P_Y_given_X = forward_action(X, W1, b1, W2, b2)
predictions = np.argmax(P_Y_given_X, axis = 1)

def classification_rate(Y, P):
    return np.mean(Y == P)

print("Score:", classification_rate(Y, predictions))



