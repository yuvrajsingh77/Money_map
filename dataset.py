import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Live Trading Dashboard", layout="wide")

st.title("📊 Live Stock Trading Dashboard")

# =========================
# LIVE CONTROLS (ADDED)
# =========================
col1, col2, col3 = st.columns(3)

if "live" not in st.session_state:
    st.session_state.live = True

with col1:
    if st.button("▶️ Start"):
        st.session_state.live = True

with col2:
    if st.button("⏸ Stop"):
        st.session_state.live = False

with col3:
    refresh_rate = st.slider("Refresh (sec)", 5, 60, 10)

# Auto refresh only when LIVE
if st.session_state.live:
    st.caption("🟢 LIVE")
    st_autorefresh(interval=refresh_rate * 1000, key="refresh")
else:
    st.caption("⏸ PAUSED")

# =========================
# USER INPUT
# =========================
ticker = st.text_input("Enter Stock", "RELIANCE.NS")

# =========================
# FETCH DATA
# =========================
df = yf.download(ticker, period="1d", interval="5m")

df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

if df.empty:
    st.error("❌ No data found")
    st.stop()

# =========================
# METRICS
# =========================
if len(df) < 2:
    st.error("❌ Not enough data")
    st.stop()

latest_price = float(df['Close'].iloc[-1])
prev_price = float(df['Close'].iloc[-2])

change = latest_price - prev_price
change_pct = (change / prev_price) * 100

high = float(df['High'].max())
low = float(df['Low'].min())
volume = int(df['Volume'].iloc[-1])

# =========================
# KPI
# =========================
col1, col2, col3, col4 = st.columns(4)

col1.metric("💰 Price", f"₹{latest_price:.2f}", f"{change:.2f} ({change_pct:.2f}%)")
col2.metric("📈 High", f"₹{high:.2f}")
col3.metric("📉 Low", f"₹{low:.2f}")
col4.metric("🔊 Volume", f"{volume:,}")

# =========================
# MOVING AVERAGES
# =========================
df['MA20'] = df['Close'].rolling(20).mean()
df['MA50'] = df['Close'].rolling(50).mean()

# =========================
# CHART
# =========================
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name='Price'
))

fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20'))
fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50'))

fig.update_layout(
    template="plotly_dark",
    height=600,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# TABLE
# =========================
st.subheader("📋 Recent Data")
st.dataframe(df.tail(20))

# =========================
# MANUAL REFRESH
# =========================
if st.button("🔄 Refresh Now"):
    st.rerun()