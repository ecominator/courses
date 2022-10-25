import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# a = np.zeros((2,3))
# b = np.ones((2,3))
# c = 10 * np.ones((2,3))
# d = np.eye(3)
# e = np.random.random()
# f = np.random.random((2,3))
# g = np.random.randn(2,3)
# R = np.random.randn(10000) #10000 element array
# Rmean = R.mean() #should be close to 0
# Rvar = R.var()  #should be close to 1
# Rstd = R.std()  #should be close to 1
# R_matrix = np.random.randn(1000,3)
# R_matrix_mean_c = R_matrix.mean(axis=0) #mean of the columns
# R_matrix_mean_r = R_matrix.mean(axis=1) #mean of the raws
# size_check = R_matrix_mean_r.shape
# covariance = np.cov(R_matrix)
# covariance_fixed = np.cov(R_matrix.T) #or np.cov(R_matrix, rowvar=False)
# h = np.random.randint(0,10, size = (3,3))
# h2 = np.random.choice(10, size = (3,3))

## second part

# A = np.array([[1, 1],[1.5, 4]])
# B = np.array([2200,5050])
# print(len(A))
# print(len(B))

# t0 = datetime.now()
# for i in range(len(A)):
#     for j in range(len(B)-1):
#         result_slow = A[i,:].dot(B[:,j])
# dt_slow = datetime.now()-t0

# t0 = datetime.now()
# result_fast = np.linalg.solve(A,B)
# dt_fast = datetime.now()-t0

# print("dt_manual/dt_numpy:", dt_slow.total_seconds()/dt_fast.total_seconds())
# print(result_slow)
# print(result_fast)

# # third part
# X = np.random.randn(200,2)
# X[:50] += 3
# Y = np.zeros(200)
# Y[:50] = 1
# plt.scatter(X[:,0],X[:,1], c=Y)
# plt.show()

# H = np.random.randn(1000)
# plt.hist(H, bins=50)
# plt.show()

# # reading image
# from PIL import Image
# im = Image.open('ex_img.png')
# #type(im)
# arr = np.array(im)
# #arr.shape
# plt.imshow(arr)
# plt.show()

# gray = arr.mean(axis=2)
# #gray.shape
# plt.imshow(gray)
# plt.show()

# plt.imshow(gray, cmap='gray')
# plt.show()


# # exercise for matplotlib

#pandas
import pandas as pd
df = pd.read_csv('data_1d.csv')
# or from a url
# !wget paste_the_url in a notebook
# df = pd.read_csv('paste_the_url')
a = df.head(10) #first 10 rows
#print(a)
# in a notebook this prints the file: !head data_1d.csv
i = df.info() 
print(i)
