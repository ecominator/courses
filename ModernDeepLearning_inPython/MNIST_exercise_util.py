import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def getData():
    # images are 28x28 = 784 size vectors
    df = pd.read_csv('digit-recognizer/train.csv')
    data = df.to_numpy().astype(np.float32)
    np.random.shuffle(data)
    X = data[:, 1:]
    Y = data[:, 0]
    
    Xtrain = X[:-1000]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:]
    Ytest = Y[-1000:]

    #normalize the data
    mu = Xtrain.mean(axis=0)
    std = Xtrain.std(axis=0)

    idx = np.where(std == 0)[0]
    assert(np.all(std[idx] == 0))

    np.place(std, std == 0, 1)
    Xtrain = (Xtrain-mu) / std
    Xtest = (Xtest-mu) / std

    return Xtrain , Xtest, Ytrain, Ytest

def y2indicator(y):
    N = len(y)
    y = y.astype(np.int32)
    K = y.max() + 1
    ind = np.zeros((N, K))
    for n in range(N):
        k = y[n]
        ind[n, k] = 1 
    return ind

def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis = 1, keepdims=True)

def forward(X, W, b):
    Z  = X.dot(W) + b 
    return softmax(Z)

def predict(P_Y_given_X):
    return np.argmax(P_Y_given_X, axis = 1) 

def error_rate(p_y, t):
    prediction = predict(p_y)
    return np.mean(prediction != t)

def cost(p_y, t):
    tot = t*np.log(p_y)
    return -np.sum(tot)

def gradW(t, y, X):
    return X.T.dot(t-y)

def gradb(t, y):
    return (t-y).sum(axis=0)
