import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="Live Stock Dashboard", layout="wide")
st.title("📈 Multi-Stock Prediction Dashboard (Live Prices)")

# ---------------- AUTO REFRESH ----------------
# Refresh every 1 second
st_autorefresh(interval=100000, key="datarefresh")

# ---------------- SIDEBAR ----------------
stocks = st.sidebar.multiselect(
    "Select Stocks",
    ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"],
    default=["RELIANCE.NS"]
)

# ---------------- FUNCTIONS ----------------
def get_live_price(ticker):
    try:
        t = yf.Ticker(ticker)
        price = t.info.get('currentPrice')
        return price
    except:
        return None

def analyze_stock(ticker):
    try:
        df = yf.download(ticker, start="2024-01-01", end=datetime.today())
        if df.empty:
            return None, "No data"

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # FEATURES
        df['ma8'] = df['Close'].rolling(8).mean()
        df['ma15'] = df['Close'].rolling(15).mean()
        df['return'] = df['Close'].pct_change() * 100
        df['volatile'] = df['return'].rolling(10).std()
        df['lag1'] = df['Close'].shift(1)
        df['lag2'] = df['Close'].shift(2)
        df['lag3'] = df['Close'].shift(3)
        df = df.bfill().ffill()

        # KMEANS
        cluster_features = ['ma8', 'ma15', 'return', 'volatile']
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(df[cluster_features])
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(x_scaled)

        # MODEL
        df['target'] = df['Close'].shift(-1)
        df = df.dropna()
        if len(df) < 50:
            return None, "Not enough data"

        feature_cols = ['Close', 'ma8', 'ma15', 'return', 'volatile', 'lag1', 'lag2', 'lag3', 'cluster']
        X = df[feature_cols]
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)

        last_row = scaler.transform(X.iloc[[-1]])
        next_price = model.predict(last_row)[0]

        # LIVE CURRENT PRICE
        last_price = get_live_price(ticker)
        if last_price is None:
            last_price = float(df['Close'].iloc[-1])

        change_pct = ((next_price - last_price) / last_price) * 100

        # SIGNAL
        if change_pct > 1.5:
            signal = "🟢 STRONG BUY"
        elif change_pct > 0.5:
            signal = "🟡 BUY"
        elif change_pct >= -0.5:
            signal = "⚪ HOLD"
        else:
            signal = "🔴 AVOID"

        return {
            "price": last_price,
            "pred": next_price,
            "change": change_pct,
            "rmse": rmse,
            "r2": r2,
            "signal": signal
        }, None

    except Exception as e:
        return None, str(e)

def plot_candles(ticker):
    # last 1 day, 1-minute interval
    df = yf.download(ticker, period="1d", interval="1m")
    if df.empty:
        return None

    df['MA8'] = df['Close'].rolling(8).mean()
    df['MA15'] = df['Close'].rolling(15).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA8'], line=dict(width=1.5), name="MA 8"))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA15'], line=dict(width=1.5), name="MA 15"))

    fig.update_layout(template="plotly_dark", title=f"{ticker} Live Candlestick", height=450, xaxis_rangeslider_visible=False)
    return fig

def plot_comparison(tickers):
    fig = go.Figure()
    for ticker in tickers:
        df = yf.download(ticker, period="7d", interval="5m")
        if df.empty:
            continue
        df_norm = df['Close'] / df['Close'].iloc[0]
        fig.add_trace(go.Scatter(x=df.index, y=df_norm, mode='lines', name=ticker))
    fig.update_layout(template="plotly_dark", title="📊 Stock Comparison (Normalized)", height=500)
    return fig

# ---------------- MAIN ----------------
for stock in stocks:
    st.subheader(f"📊 {stock}")
    result, error = analyze_stock(stock)
    if error:
        st.error(f"{stock}: {error}")
        continue

    col1, col2, col3 = st.columns(3)
    col1.metric("Last Price", f"₹{result['price']:.2f}")
    col2.metric("Predicted", f"₹{result['pred']:.2f}")
    col3.metric("Change %", f"{result['change']:.2f}%")
    st.write(f"**Suggestion:** {result['signal']}")
    st.write(f"RMSE: {result['rmse']:.2f} | R²: {result['r2']:.3f}")

    st.plotly_chart(plot_candles(stock), use_container_width=True)
    st.divider()

# ---------------- COMPARISON ----------------
if stocks:
    st.subheader("📊 Stock Comparison")
    st.plotly_chart(plot_comparison(stocks), use_container_width=True)