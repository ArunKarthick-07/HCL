import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

housing = fetch_california_housing()
X = housing.data
y = housing.target

X_simple = X[:, 0].reshape(-1, 1) 

X_train_simple, X_test_simple, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)

simple_lr = LinearRegression()
simple_lr.fit(X_train_simple, y_train)

y_pred_simple = simple_lr.predict(X_test_simple)
mse_simple = mean_squared_error(y_test, y_pred_simple)
r2_simple = r2_score(y_test, y_pred_simple)
print(f"Simple Linear Regression - MSE: {mse_simple:.2f}, R2: {r2_simple:.2f}")

X_train_multi, X_test_multi, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

multi_lr = LinearRegression()
multi_lr.fit(X_train_multi, y_train)


y_pred_multi = multi_lr.predict(X_test_multi)
mse_multi = mean_squared_error(y_test, y_pred_multi)
r2_multi = r2_score(y_test, y_pred_multi)
print(f"Multiple Linear Regression - MSE: {mse_multi:.2f}, R2: {r2_multi:.2f}")