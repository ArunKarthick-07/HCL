import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


stock_data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')
data = stock_data['Close'].values.reshape(-1, 1)


df = pd.DataFrame(data, columns=['Close'])
df['Lag1'] = df['Close'].shift(1)
df.dropna(inplace=True)

X = df[['Lag1']]
y = df['Close']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()