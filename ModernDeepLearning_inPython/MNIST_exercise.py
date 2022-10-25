import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from MNIST_exercise_util import getData, y2indicator, forward, error_rate, cost, gradW, gradb

# def one_neuron_benchmark():
#     Xtrain, Ytrain, Xtest, Ytest = getData()
    
#     N, D = Xtrain.shape #N: num of samples, D: num of features
#     # for multi-class classification, turn the targets into indicator matrices 
#     Ytrain_ind = y2indicator(Ytrain)
#     Ytest_ind = y2indicator(Ytest)
#     K = Ytrain_ind.shape[1] # num of columns in the indicator matrix
    

#     W = np.random.randn(D,10) / np.sqrt(D)
#     b = np.zeros(10)
#     LL = []
#     LLtest = []
#     CRtest = []

#     learning_rate = 0.00004
#     reg = 0.01 # regularization penalty
#     for i in range(500):
#         p_y = forward(Xtrain, W, b)
#         # print "p_y:", p_y
#         ll = cost(p_y, Ytrain_ind)
#         LL.append(ll)

#         p_y_test = forward(Xtest, W, b)
#         lltest = cost(p_y_test, Ytest_ind)
#         LLtest.append(lltest)

#         err = error_rate(p_y_test, Ytest)
#         CRtest.append(err)

#         W += learning_rate*(gradW(Ytrain_ind, p_y, Xtrain) - reg*W)
#         b += learning_rate*(gradb(Ytrain_ind, p_y) - reg*b)

#         if i % 10 == 0:
#             print ("Cost at iteration %d: %.5f", i, ll )
#             print ("Error rate:", err)
        
#     p_y = forward(Xtest, W, b)
#     print("Final error rate:" , error_rate(p_y, Ytest) )
#     iters = range(len(LL))
    
#     plt.plot(iters, LL, iters, LLtest)
#     plt.show()
#     plt.plot(CRtest)
#     plt.show()

def one_neuron_benchmark():
    Xtrain, Xtest, Ytrain, Ytest = getData()

    print("Performing logistic regression...")
    # lr = LogisticRegression(solver='lbfgs')


    # convert Ytrain and Ytest to (N x K) matrices of indicator variables
    N, D = Xtrain.shape
    # for multi-class classification, turn the targets into indicator matrices 
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    W = np.random.randn(D, 10) / np.sqrt(D)
    b = np.zeros(10)
    LL = []
    LLtest = []
    CRtest = []

    lr = 0.00004
    reg = 0.01
    for i in range(500):
        p_y = forward(Xtrain, W, b)
        # print "p_y:", p_y
        ll = cost(p_y, Ytrain_ind)
        LL.append(ll)

        p_y_test = forward(Xtest, W, b)
        lltest = cost(p_y_test, Ytest_ind)
        LLtest.append(lltest)
        
        err = error_rate(p_y_test, Ytest)
        CRtest.append(err)

        W += lr*(gradW(Ytrain_ind, p_y, Xtrain) - reg*W)
        b += lr*(gradb(Ytrain_ind, p_y) - reg*b)
        if i % 10 == 0:
            print("Cost at iteration %d: %.6f" % (i, ll))
            print("Error rate:", err)

    p_y = forward(Xtest, W, b)
    print("Final error rate:", error_rate(p_y, Ytest))
    iters = range(len(LL))
    plt.plot(iters, LL, iters, LLtest)
    plt.show()
    plt.plot(CRtest)
    plt.show()


if __name__ == '__main__':
    one_neuron_benchmark()