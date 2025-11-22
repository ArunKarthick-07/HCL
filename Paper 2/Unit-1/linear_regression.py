import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# Load dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Linear Regression ---
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

# --- Ridge Regression ---
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
pred_ridge = ridge.predict(X_test)

# --- Lasso Regression ---
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
pred_lasso = lasso.predict(X_test)

# Performance Comparison
print("==== MODEL PERFORMANCE ====")
print("\nLinear Regression:")
print("MSE:", mean_squared_error(y_test, pred_lr))
print("R2 Score:", r2_score(y_test, pred_lr))

print("\nRidge Regression:")
print("MSE:", mean_squared_error(y_test, pred_ridge))
print("R2 Score:", r2_score(y_test, pred_ridge))

print("\nLasso Regression:")
print("MSE:", mean_squared_error(y_test, pred_lasso))
print("R2 Score:", r2_score(y_test, pred_lasso))
