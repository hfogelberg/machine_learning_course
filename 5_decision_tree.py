# Decision Tree Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fit decision tree regressor to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x, y)

# Predict result
y_pred = regressor.predict(6.5)
print(y_pred)

# Visualize result
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color='blue')
plt.title("Decision Tree Regression")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

