import numpy as np
import matplotlib.pyplot as plt

# load the data
X = []
Y = []

for line in open ('data_1d.csv'):
    x,y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

# turn lists to numpy arrays
X = np.array(X)
Y = np.array(Y)

# plot the data to see what it looks like
plt.scatter(X,Y)
plt.show()

# calculate the constants
denominator = X.dot(X) - X.mean() * X.sum()
a = ( X.dot(Y) - Y.mean() * X.sum() ) / denominator
b = ( Y.mean() * X.dot(X) - X.mean()*X.dot(Y) ) / denominator
 
# calculate the predicted Y
Yhat = a*X + b

# plot it all
plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

# r-squared

d1 = Y-Yhat
d2 = Y-Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("the r-squared result is:", r2)
