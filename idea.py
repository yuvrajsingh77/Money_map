
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

ticker = "AAPL"
df = yf.download(ticker, start="2020-01-01", end="2025-01-01")
df.columns = [col[0] for col in df.columns]

df['MA7']           = df['Close'].rolling(window=7).mean()
df['MA21']          = df['Close'].rolling(window=21).mean()
df['Daily_Return']  = df['Close'].pct_change() * 100
df['Volatility']    = df['Daily_Return'].rolling(window=7).std()
df['Lag1']          = df['Close'].shift(1)
df['Lag2']          = df['Close'].shift(2)
df['Lag3']          = df['Close'].shift(3)
df['Lag5']          = df['Close'].shift(5)
df['Price_Change1'] = df['Close'] - df['Lag1']
df['Price_Change3'] = df['Close'] - df['Lag3']
df.dropna(inplace=True)

cluster_features = ['Daily_Return', 'Volatility', 'MA7', 'MA21']
cluster_scaler = StandardScaler()
X_cluster = cluster_scaler.fit_transform(df[cluster_features])
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_cluster)

df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

feature_cols = [
    'Close', 'MA7', 'MA21', 'Daily_Return', 'Volatility',
    'Lag1', 'Lag2', 'Lag3', 'Lag5',
    'Price_Change1', 'Price_Change3', 'Cluster'
]
X = df[feature_cols]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae  = mean_absolute_error(y_test, predictions)
r2   = r2_score(y_test, predictions)

print("===== Model Performance =====")
print(f"RMSE     : {rmse:.2f}")
print(f"MAE      : {mae:.2f}")
print(f"R² Score : {r2:.4f}")

last_row   = scaler.transform(X.iloc[[-1]])
next_price = model.predict(last_row)[0]
last_price = df['Close'].iloc[-1]
change     = next_price - last_price

print(f"\nLast closing price   : ${last_price:.2f}")
print(f"Predicted next close : ${next_price:.2f}")
print(f"Expected change      : ${change:+.2f}")

plt.figure(figsize=(12, 5))
plt.plot(y_test.values, label='Actual price',    color='blue', linewidth=1.5)
plt.plot(predictions,   label='Predicted price', color='red',  linewidth=1.5, linestyle='--')
plt.title("AAPL — Actual vs Predicted (Linear Regression)")
plt.xlabel("Days (test set)")
plt.ylabel("Price ($)")
plt.legend()
plt.tight_layout()
plt.show()