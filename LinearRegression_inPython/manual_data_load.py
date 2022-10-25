# data loading
import numpy as np

X = []
for line in open('path of the file'): #path can be smtg blike ..\previous_folder\folder_that_file_is_located\file_itself.csv
    row = line.split(',')
    sample = map(float, row)
    X.append(sample)

# convert to numpy array
X = np.array(X)
# print(X)