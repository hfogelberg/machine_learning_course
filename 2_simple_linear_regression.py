# Simple Linear Regression

# Importing the libraries
import numpy
import matplotlib.pyplot
import pandas
from sklearn.linear_model import LinearRegression

# import the dataset
dataset = pandas.read_csv('Salary_Data.csv')
matrix = dataset.iloc[:, :-1].values
salary = dataset.iloc[:, 1].values

# Split dataset between Traing set and Test set
matrix_train, matrix_test, salary_train, salary_test = train_test_split(matrix, salary, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(matrix_train, salary_train)

# Predict the Test set result
predictions = regressor.predict(matrix_test)
print("Predictions")
print(predictions)

