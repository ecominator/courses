import numpy as np
import pandas as pd
#import os

#cwd = os.getcwd()
#print('cws is:',cwd)
#df = pd.read_csv(f'{cwd}/Desktop/ann_logistic_extra/ecommerce_data.csv', delimiter = ',')
#print(df.head()) #to see what's inside the csv

def get_data():
    df = pd.read_csv('./Desktop/ann_logistic_extra/ecommerce_data.csv')
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1].astype(np.int32)

    #normalize the numerical columns
    X[:,1] = (X[:,1]-X[:,1].mean()) / X[:, 1].std()
    X[:,2] = (X[:,2]-X[:,2].mean()) / X[:, 2].std()

    #time of the day column
    N, D = X.shape
    X2 = np.zeros((N,D+3)) #bcs there are 4 different categorical values which are the times of the day
    X2[:, 0:(D-1)] = X[:, 0:(D-1)]
    
    #one-hot encoding for the times of the day
    for n in range(N):
        t = int(X[n, D-1]) #this is time of the day: 0,1,2 or 3
        X2[n, t+D-1] = 1
    
    # Z = np.zeros((N, 4))
    # Z[np.arange(N),X[:,D-1].astype(np.int32)] = 1
    # X2[:, -4:] = Z
    # assert(np.abs(X2[:,-4:]-Z).sum()<1e-10)

    return X2, Y

def get_binary_data():
    X, Y = get_data()
    X2 = X[Y<=1]
    Y2 = Y[Y<=1]
    return X2, Y2

