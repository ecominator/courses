import numpy as np
from sklearn.datasets import load_breast_cancer

# load the data
data  = load_breast_cancer()

# check the data type
print(type(data))
print(data.keys())
#print(data.data.shape)

#print(data.target)
#print(data.target_names)
#print(data.target.shape)

#print(data.feature_names)
#print(data.feature_names.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)

# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier()
# model.fit(X_train, y_train)

# # evaluate the model's performance
# print(model.score(X_train, y_train))
# print(model.score(X_test, y_test))

# # how to make predictions
# prediction = model.predict(X_test)
# print(prediction)

# N = len(y_test)
# accuracy = np.sum(prediction==y_test)/N # or np.mean(prediction==y_test)
# print(accuracy)

# use deep learning to solve the same problem
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train2 = scaler.fit_transform(X_train)
X_test2 = scaler.transform(X_test)

model = MLPClassifier(max_iter=500)
model.fit(X_train2, y_train)

# evaluate the model's performance
print(model.score(X_train2, y_train))
print(model.score(X_test2, y_test))
