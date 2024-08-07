from sklearn.model_selection import train_test_split
from sklearn. linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import numpy as np

x = np.array([1,2,3,4,5,6,7,8,9,10]). reshape(-1,1)
y = np.array([0,1,1,2,3,5,8,13,21,34])

poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(x)
print(x_poly)

x_train, x_test, y_train, y_test = train_test_split(x_poly, y, test_size=0.25, random_state=8)

model = LinearRegression()

model.fit(x_train, y_train)

y_pred = model. predict(x_test)

print(y_pred, y_test)

mse = mean_squared_error(y_test, y_pred)

print(mse)
