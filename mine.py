import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mplfinance as mpf

ticker_name = "TSLA"

# =====================================================
# STEP 1: GET DATA
# =====================================================

df_historical = yf.download(ticker_name, start="2022-01-01", end=date.today(), progress=False)
df_historical.columns = [col[0] for col in df_historical.columns]

df_live = yf.download(ticker_name, period="1d", interval="1m", progress=False)

if len(df_live) == 0:
    print("Market is closed or no live data. Using last available price.")
    df = df_historical.copy()
else:
    df_live.columns = [col[0] for col in df_live.columns]
    df_live = df_live[['Open', 'High', 'Low', 'Close', 'Volume']]
    df_live.index = df_live.index.tz_localize(None)

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

    df = pd.concat([df_historical, today_row])
    df = df[~df.index.duplicated(keep='last')]

df.sort_index(inplace=True)

print(f"Total rows : {len(df)}")
print(f"Last price : ₹{float(df['Close'].iloc[-1]):.2f}")

# =====================================================
# STEP 2: FEATURE ENGINEERING (20 FEATURES)
# =====================================================

# --- price & lag (4) ---
df['return'] = df['Close'].pct_change() * 100
df['lag1']   = df['Close'].shift(1)
df['lag2']   = df['Close'].shift(2)
df['lag3']   = df['Close'].shift(3)

# --- moving averages (3) ---
df['ma8']   = df['Close'].rolling(window=8).mean()
df['ma15']  = df['Close'].rolling(window=15).mean()
df['ema21'] = df['Close'].ewm(span=21, adjust=False).mean()

# --- volatility (3) ---
df['volatile'] = df['return'].rolling(window=10).std()
bb_mid         = df['Close'].rolling(window=20).mean()
bb_std         = df['Close'].rolling(window=20).std()
df['bb_upper'] = bb_mid + (2 * bb_std)
df['bb_lower'] = bb_mid - (2 * bb_std)

# --- momentum (4) ---
delta       = df['Close'].diff()
gain        = delta.clip(lower=0).rolling(window=14).mean()
loss        = (-delta.clip(upper=0)).rolling(window=14).mean()
df['rsi14'] = 100 - (100 / (1 + gain / loss))
df['roc10'] = df['Close'].pct_change(periods=10) * 100
df['mom10'] = df['Close'] - df['Close'].shift(10)

highest_high  = df['High'].rolling(window=14).max()
lowest_low    = df['Low'].rolling(window=14).min()
df['stoch_k'] = ((df['Close'] - lowest_low) / (highest_high - lowest_low)) * 100

# --- trend (3) ---
ema12             = df['Close'].ewm(span=12, adjust=False).mean()
ema26             = df['Close'].ewm(span=26, adjust=False).mean()
df['macd']        = ema12 - ema26
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

tr        = pd.concat([
    df['High'] - df['Low'],
    (df['High'] - df['Close'].shift()).abs(),
    (df['Low']  - df['Close'].shift()).abs()
], axis=1).max(axis=1)
df['atr'] = tr.rolling(window=14).mean()

# --- volume (2) ---
df['vol_ma10']  = df['Volume'].rolling(window=10).mean()
df['vol_ratio'] = df['Volume'] / df['vol_ma10']

# --- price pattern (1) ---
df['hl_range'] = df['High'] - df['Low']

df.dropna(inplace=True)
print(f"Rows after cleaning: {len(df)}")

# =====================================================
# STEP 3: K-MEANS CLUSTERING
# =====================================================

cluster_features = ['ma8', 'ma15', 'return', 'volatile',
                    'rsi14', 'macd', 'atr', 'vol_ratio']
scaler    = StandardScaler()
x_scaled  = scaler.fit_transform(df[cluster_features])

kmeans        = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(x_scaled)

print("\nDays per cluster:")
print(df['cluster'].value_counts())

# =====================================================
# STEP 4: SUPERVISED MODEL
# =====================================================

df['target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

feature_cols = [
    # price & lag
    'Close', 'return', 'lag1', 'lag2', 'lag3',
    # moving averages
    'ma8', 'ma15', 'ema21',
    # volatility
    'volatile', 'bb_upper', 'bb_lower',
    # momentum
    'rsi14', 'roc10', 'mom10', 'stoch_k',
    # trend
    'macd', 'macd_signal', 'atr',
    # volume
    'vol_ratio',
    # cluster
    'cluster'
]

print(f"\nTotal features used: {len(feature_cols)}")

X = df[feature_cols]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

feature_scaler = MinMaxScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled  = feature_scaler.transform(X_test)

model = Ridge(alpha=10.0)
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)

# =====================================================
# STEP 5: EVALUATE
# =====================================================

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae  = mean_absolute_error(y_test, predictions)
r2   = r2_score(y_test, predictions)

train_preds = model.predict(X_train_scaled)
train_r2    = r2_score(y_train, train_preds)

print("\n===== Model Performance =====")
print(f"RMSE       : {rmse:.2f}")
print(f"MAE        : {mae:.2f}")
print(f"Train R²   : {train_r2:.4f}")
print(f"Test  R²   : {r2:.4f}")
print(f"Difference : {train_r2 - r2:.4f}  {'(overfitting)' if train_r2 - r2 > 0.05 else '(healthy)'}")

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