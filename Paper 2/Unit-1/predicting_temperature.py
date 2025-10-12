import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
n_samples = len(dates)
data = pd.DataFrame({
    'Date': dates,
    'Temp': 15 + 10 * np.sin(np.arange(n_samples) * 2 * np.pi / 365) + np.random.normal(0, 2, n_samples),
    'Humidity': np.random.uniform(30, 90, n_samples),
    'Pressure': np.random.uniform(990, 1030, n_samples)
})
data.set_index('Date', inplace=True)

data['Lag1'] = data['Temp'].shift(1)
data['Lag2'] = data['Temp'].shift(2)
data.dropna(inplace=True)

X = data[['Lag1', 'Lag2', 'Humidity', 'Pressure']]
y = data['Temp']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error for Temperature: {mse:.2f}")

plt.plot(y_test.values[:100], label='Actual')
plt.plot(y_pred[:100], label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Temperature')
plt.title('Temperature Prediction')
plt.legend()
plt.show()