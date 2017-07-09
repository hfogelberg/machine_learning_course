# Polynomial Regression


# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
ds = pd.read_csv("Position_Salaries.csv")
x = ds.iloc[:, 1:2].values   # Independent variables
y = ds.iloc[:, 2].values     # Dependent variable

print(x)
print(y)

# Linear regression model as reference
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
res  = linreg.fit(x, y)

# Polynomial regresion model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly, y)
linreg2 = LinearRegression()
linreg2.fit(x_poly, y)

# Dump to file
f = open('x_poly', 'w+')
f.write(str(x_poly))
f.close()

# Visualize linear regression results
# plt.scatter(x, y, color = 'red')
# plt.plot(x, linreg.predict(x), color = 'blue')
# plt.title('Truth or bluf (Linear Regressions')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()

# Visualize polynomial regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, linreg2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('Truth or bluf (Polynomial Regressions')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


