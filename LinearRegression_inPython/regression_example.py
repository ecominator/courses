import numpy as np
import pandas as pd

df = pd.read_csv('airfoil_self_noise.dat', sep='\t',header=None)
# print(df.head())
# print(df.info())

# get the inputs
data = df[[0,1,2,3,4]].values

# get the outputs
target = df[5].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

predictions = model.predict(X_test)
#print(predictions)

# random forest to compare
from sklearn.ensemble import RandomForestRegressor
model2 = RandomForestRegressor()
model2.fit(X_train, y_train)

print(model2.score(X_train, y_train))
print(model2.score(X_test, y_test))