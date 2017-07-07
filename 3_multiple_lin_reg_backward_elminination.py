# Multiple Linear Regression

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
ds = pd.read_csv("50_Startups.csv")
x = ds.iloc[:, :-1].values   # Independent variables
y = ds.iloc[:, 4].values     # Dependent variable

# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
le_x = LabelEncoder()
x[:, 3] = le_x.fit_transform(x[:, 3])
ohc = OneHotEncoder(categorical_features = [3])
x = ohc.fit_transform(x).toarray()

# Avoiding the Dummy Data Trap
x = x[:, 1:]

# Split dataset into Training set and Testing set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# Split dataset into Training set and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# Fit Multiple Linear Regression to Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Prediction of Test set Results
y_pred = regressor.predict(x_test)

# print("Y-test")
# print(y_test)
# print("Y-pred")
# print(y_pred)

# Building optimal model with backward elimination
import statsmodels.formula.api as sm

x = np.append(arr = np.ones((50, 1)).astype(int), values = x, axis = 1)     # Must insert constant at start of dataset to fit linear formula

print(x)