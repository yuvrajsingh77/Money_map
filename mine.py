import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mplfinance as mpf

ticker_name = "RELIANCE.NS"

# =====================================================
# STEP 1: GET DATA
# =====================================================

# historical daily data
df_historical = yf.download(ticker_name, start="2024-01-01", end=date.today(), progress=False)
df_historical.columns = [col[0] for col in df_historical.columns]

# live 1-minute data for today's real price
df_live = yf.download(ticker_name, period="1d", interval="1m", progress=False)
df_live.columns = [col[0] for col in df_live.columns]
df_live = df_live[['Open', 'High', 'Low', 'Close', 'Volume']]
df_live.index = df_live.index.tz_localize(None)

# build today's candle manually from live data
today_open   = float(df_live['Open'].iloc[0])
today_high   = float(df_live['High'].max())
today_low    = float(df_live['Low'].min())
today_close  = float(df_live['Close'].iloc[-1])
today_volume = int(df_live['Volume'].sum())

today_row = pd.DataFrame({
    'Open':   [today_open],
    'High':   [today_high],
    'Low':    [today_low],
    'Close':  [today_close],
    'Volume': [today_volume]
}, index=[pd.Timestamp(date.today())])

# combine historical + today
df = pd.concat([df_historical, today_row])
df = df[~df.index.duplicated(keep='last')]
df.sort_index(inplace=True)

print(f"Total rows : {len(df)}")
print(f"Last price : ₹{today_close:.2f}")

# =====================================================
# STEP 2: FEATURE ENGINEERING
# =====================================================

df['ma8']      = df['Close'].rolling(window=8).mean()
df['ma15']     = df['Close'].rolling(window=15).mean()
df['return']   = df['Close'].pct_change() * 100
df['volatile'] = df['return'].rolling(window=10).std()
df['lag1']     = df['Close'].shift(1)
df['lag2']     = df['Close'].shift(2)
df['lag3']     = df['Close'].shift(3)

df.dropna(inplace=True)
print(f"Rows after cleaning: {len(df)}")

# =====================================================
# STEP 3: K-MEANS CLUSTERING
# =====================================================

cluster_features = ['ma8', 'ma15', 'return', 'volatile']
scaler = StandardScaler()
x_scaled = scaler.fit_transform(df[cluster_features])

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(x_scaled)

print("\nDays per cluster:")
print(df['cluster'].value_counts())

# =====================================================
# STEP 4: SUPERVISED MODEL
# =====================================================

df['target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

feature_cols = ['Close', 'ma8', 'ma15', 'return',
                'volatile', 'lag1', 'lag2', 'lag3', 'cluster']

X = df[feature_cols]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

feature_scaler = MinMaxScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled  = feature_scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)

# =====================================================
# STEP 5: EVALUATE
# =====================================================

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae  = mean_absolute_error(y_test, predictions)
r2   = r2_score(y_test, predictions)

print("\n===== Model Performance =====")
print(f"RMSE     : {rmse:.2f}")
print(f"MAE      : {mae:.2f}")
print(f"R² Score : {r2:.4f}")

last_row   = feature_scaler.transform(X.iloc[[-1]])
next_price = model.predict(last_row)[0]
last_price = float(df['Close'].iloc[-1])
change     = next_price - last_price
change_pct = (change / last_price) * 100

print(f"\nLast known price     : ₹{last_price:.2f}")
print(f"Predicted next price : ₹{next_price:.2f}")
print(f"Expected change      : ₹{change:+.2f} ({change_pct:+.2f}%)")

# =====================================================
# STEP 6: BUY / SELL / HOLD SIGNAL
# =====================================================

current_volatility = float(df['volatile'].iloc[-1])
avg_volatility     = float(df['volatile'].mean())

print("\n===== Trading Signal =====")

if change_pct > 1.5:
    signal = "STRONG BUY"
    advice = "Model predicts a solid upward move. Good time to enter."
elif change_pct > 0.5:
    signal = "BUY"
    advice = "Model expects moderate growth. Consider buying."
elif change_pct >= -0.5:
    signal = "HOLD / WAIT"
    advice = "Predicted change is too small to act on. Wait for a clearer move."
else:
    signal = "AVOID / DON'T BUY"
    advice = "Model predicts a price drop. Better to stay out."

print(f"Signal : {signal}")
print(f"Advice : {advice}")

# =====================================================
# STEP 7: RISK ASSESSMENT
# =====================================================

print("\n===== Risk Assessment =====")

if current_volatility > avg_volatility * 1.5:
    risk_level = "HIGH RISK"
    risk_note  = "Market is unusually volatile. Even a Buy signal carries extra risk."
elif current_volatility > avg_volatility * 1.0:
    risk_level = "MODERATE RISK"
    risk_note  = "Volatility is slightly above average. Proceed with caution."
else:
    risk_level = "LOW RISK"
    risk_note  = "Market is relatively stable. Signal is more reliable."

print(f"Risk Level        : {risk_level}")
print(f"Current Volatility: {current_volatility:.4f}")
print(f"Average Volatility: {avg_volatility:.4f}")
print(f"Note              : {risk_note}")

# =====================================================
# STEP 8: COMBINED VERDICT
# =====================================================

print("\n===== Combined Verdict =====")

if "BUY" in signal and "LOW" in risk_level:
    verdict = "Best case scenario — good opportunity in a stable market. Strong entry point."
elif "BUY" in signal and "HIGH" in risk_level:
    verdict = "Upward move expected but market is volatile. Buy small or wait for calm."
elif "BUY" in signal and "MODERATE" in risk_level:
    verdict = "Moderate opportunity with slightly elevated risk. Consider a small position."
elif "HOLD" in signal and "LOW" in risk_level:
    verdict = "Market is stable but no strong move predicted. Watch and wait."
elif "HOLD" in signal and "HIGH" in risk_level:
    verdict = "No clear opportunity and market is volatile. Stay out for now."
elif "AVOID" in signal:
    verdict = "Price drop expected. Do not enter regardless of risk level."
else:
    verdict = "Monitor closely before making a decision."

print(f"Verdict : {verdict}")

# =====================================================
# STEP 9: CANDLESTICK CHART
# =====================================================

df_candle = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(90).copy()
df_candle.index = pd.to_datetime(df_candle.index)

ap = [
    mpf.make_addplot(df_candle['Close'].rolling(8).mean(),
                     color='orange', width=1.5, label='MA8'),
    mpf.make_addplot(df_candle['Close'].rolling(15).mean(),
                     color='purple', width=1.5, label='MA15'),
]

mpf.plot(df_candle,
         type='candle',
         style='charles',
         title=f'{ticker_name} — Last 90 Days',
         ylabel='Price (₹)',
         volume=True,
         addplot=ap,
         figsize=(12, 6))