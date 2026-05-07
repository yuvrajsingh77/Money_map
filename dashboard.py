import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="QuantAI — Stock Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Outfit:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif; background-color: #060810; color: #C8D0E0; }
.main { background-color: #060810; }
.block-container { padding: 1.2rem 2rem 2rem 2rem; max-width: 1800px; }
[data-testid="stSidebar"] { background: #080B14; border-right: 1px solid #141C2E; }
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

.hero-wrap {
    background: linear-gradient(135deg, #0B0F1E 0%, #0D1528 60%, #0B1220 100%);
    border: 1px solid #1A2640; border-radius: 20px;
    padding: 2rem 2.5rem 1.8rem; margin-bottom: 1.4rem;
    position: relative; overflow: hidden;
}
.hero-wrap::before {
    content: ''; position: absolute; top: -120px; right: -120px;
    width: 380px; height: 380px;
    background: radial-gradient(circle, rgba(99,179,237,0.06) 0%, transparent 65%);
}
.hero-eyebrow { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; color: #4A90D9; letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 0.5rem; }
.hero-title { font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 2.2rem; color: #EDF2FF; margin: 0 0 0.4rem 0; letter-spacing: -0.03em; line-height: 1.1; }
.hero-title span { color: #63B3ED; }
.hero-badges { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 0.8rem; }
.hero-badge { display: inline-block; background: rgba(99,179,237,0.08); border: 1px solid rgba(99,179,237,0.2); border-radius: 6px; padding: 0.22rem 0.75rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #63B3ED; }
.hero-badge.green { background: rgba(72,187,120,0.08); border-color: rgba(72,187,120,0.2); color: #48BB78; }
.hero-badge.purple { background: rgba(154,102,255,0.08); border-color: rgba(154,102,255,0.2); color: #9A66FF; }
.hero-badge.gold { background: rgba(251,191,36,0.08); border-color: rgba(251,191,36,0.2); color: #FBBF24; }

.kpi-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.85rem; margin-bottom: 1.2rem; }
.kpi-card { background: #0B0F1E; border: 1px solid #141C2E; border-radius: 16px; padding: 1.1rem 1.3rem 1rem; position: relative; overflow: hidden; transition: border-color 0.25s, transform 0.2s; }
.kpi-card:hover { border-color: #2A4A7A; transform: translateY(-2px); }
.kpi-accent { position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: 16px 16px 0 0; }
.kpi-accent.blue   { background: linear-gradient(90deg, #2563EB, #63B3ED); }
.kpi-accent.green  { background: linear-gradient(90deg, #059669, #48BB78); }
.kpi-accent.red    { background: linear-gradient(90deg, #DC2626, #F87171); }
.kpi-accent.purple { background: linear-gradient(90deg, #7C3AED, #9A66FF); }
.kpi-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.58rem; color: #4B5A72; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.55rem; margin-top: 0.3rem; }
.kpi-val { font-family: 'Outfit', sans-serif; font-size: 1.55rem; font-weight: 700; line-height: 1; color: #EDF2FF; }
.kpi-val.up { color: #48BB78; } .kpi-val.down { color: #F87171; } .kpi-val.blue { color: #63B3ED; } .kpi-val.purple { color: #9A66FF; }
.kpi-sub { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; color: #4B5A72; margin-top: 0.3rem; }
.kpi-sub.up { color: #48BB78; } .kpi-sub.down { color: #F87171; }

.signal-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.85rem; margin-bottom: 1.2rem; }
.signal-card { border-radius: 16px; padding: 1.1rem 1.4rem; border: 1px solid; display: flex; align-items: center; gap: 1rem; }
.signal-card.strong-buy { background: rgba(5,150,105,0.07); border-color: rgba(5,150,105,0.3); }
.signal-card.buy        { background: rgba(72,187,120,0.07); border-color: rgba(72,187,120,0.3); }
.signal-card.hold       { background: rgba(217,119,6,0.07);  border-color: rgba(217,119,6,0.3); }
.signal-card.avoid      { background: rgba(220,38,38,0.07);  border-color: rgba(220,38,38,0.3); }
.signal-icon { font-size: 2rem; line-height: 1; }
.signal-lbl { font-family: 'IBM Plex Mono', monospace; font-size: 0.58rem; color: #4B5A72; text-transform: uppercase; letter-spacing: 0.1em; }
.signal-name { font-family: 'Outfit', sans-serif; font-size: 1.4rem; font-weight: 700; margin-top: 0.05rem; }
.signal-name.strong-buy { color: #059669; } .signal-name.buy { color: #48BB78; } .signal-name.hold { color: #FBBF24; } .signal-name.avoid { color: #F87171; }
.signal-desc { font-size: 0.78rem; color: #6B7A94; margin-top: 0.1rem; }
.verdict-card { background: #0B0F1E; border: 1px solid #1A2640; border-radius: 16px; padding: 1.1rem 1.4rem; }
.risk-card    { background: #0B0F1E; border: 1px solid #1A2640; border-radius: 16px; padding: 1.1rem 1.4rem; }

.sec-hdr { font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem; color: #4A90D9; text-transform: uppercase; letter-spacing: 0.16em; margin-bottom: 0.8rem; padding-bottom: 0.5rem; border-bottom: 1px solid #141C2E; }
.pill-row { display: flex; flex-wrap: wrap; gap: 0.55rem; margin-bottom: 1rem; }
.pill { background: #0B0F1E; border: 1px solid #1A2640; border-radius: 8px; padding: 0.35rem 0.85rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; }
.pill span { color: #4B5A72; } .pill strong { color: #C8D0E0; margin-left: 0.35rem; }

.stTabs [data-baseweb="tab-list"] { background: #0B0F1E; border-radius: 12px; padding: 4px; border: 1px solid #141C2E; gap: 2px; }
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 9px; color: #4B5A72; font-family: 'Outfit', sans-serif; font-size: 0.83rem; font-weight: 600; padding: 0.4rem 1rem; border: none; transition: all 0.2s; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #0F1E3A, #0D1A34) !important; color: #63B3ED !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1rem; }

.mode-banner { background: rgba(99,179,237,0.06); border: 1px solid rgba(99,179,237,0.2); border-radius: 10px; padding: 0.5rem 1rem; margin-bottom: 1rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: #63B3ED; }

#MainMenu, footer, header { visibility: hidden; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #060810; }
::-webkit-scrollbar-thumb { background: #1A2640; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #2563EB; }
.stSpinner > div { border-top-color: #63B3ED !important; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("""
    <div style='font-family:IBM Plex Mono,monospace;font-size:0.85rem;color:#63B3ED;font-weight:700;margin-bottom:0.3rem;'>⚡ QuantAI</div>
    <div style='font-family:IBM Plex Mono,monospace;font-size:0.58rem;color:#2A4A7A;letter-spacing:0.12em;margin-bottom:1.2rem;'>STOCK INTELLIGENCE PLATFORM</div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("<p style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:#4B5A72;text-transform:uppercase;letter-spacing:0.1em;'>Ticker Symbol</p>", unsafe_allow_html=True)
    ticker_name = st.text_input("", value="RELIANCE.NS", label_visibility="collapsed")

    st.markdown("<p style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:#4B5A72;text-transform:uppercase;letter-spacing:0.1em;'>Start Date</p>", unsafe_allow_html=True)
    start_date = st.date_input("", value=date(2022, 1, 1), label_visibility="collapsed")

    # Show data mode hint based on date range
    days_diff = (date.today() - start_date).days
    if days_diff <= 30:
        hint_text = "📡 Mode: Hourly 60d · 20 features"
        hint_color = "#48BB78"
    elif days_diff <= 180:
        hint_text = "📡 Mode: Hourly 6mo · 20 features"
        hint_color = "#48BB78"
    elif days_diff <= 365:
        hint_text = "📡 Mode: Hourly 1yr · 12 features"
        hint_color = "#FBBF24"
    else:
        hint_text = "📡 Mode: Daily · 20 features"
        hint_color = "#63B3ED"

    st.markdown(f"""
    <div style='background:rgba(99,179,237,0.05);border:1px solid rgba(99,179,237,0.15);
                border-radius:8px;padding:0.4rem 0.7rem;margin-top:0.4rem;
                font-family:IBM Plex Mono,monospace;font-size:0.62rem;color:{hint_color};'>
    {hint_text}
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("<p style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:#4B5A72;text-transform:uppercase;letter-spacing:0.1em;'>Model Parameters</p>", unsafe_allow_html=True)
    model_alpha = st.slider("Ridge α (regularisation)", 0.1, 20.0, 10.0, step=0.1)
    n_clusters  = st.slider("KMeans Clusters (k)", 2, 6, 3)

    st.divider()
    if st.button("🔄  Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("""
    <div style='font-family:IBM Plex Mono,monospace;font-size:0.6rem;color:#1E2D45;line-height:2.2;margin-top:1rem;'>
    MODEL STACK<br>› Ridge Regression<br>› KMeans Clustering<br>› Adaptive Features<br>› Live Data via yfinance
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# BUILD MODEL — fully cached, adaptive data + features
# =========================================================

@st.cache_data(ttl=300)
def build_model(ticker, sd, alpha, k):

    today     = date.today()
    days_diff = (today - sd).days

    # ── ADAPTIVE DATA FETCH ──
    # Short range = use hourly data to get enough rows
    # Long range  = use daily data
    if days_diff <= 30:
        # up to 1 month selected → fetch 60 days hourly (~500 rows)
        df = yf.download(ticker, period="60d", interval="1h", progress=False)
        data_mode = "Hourly · 60d window"
    elif days_diff <= 180:
        # 1–6 months → fetch 6 months hourly (~700 rows)
        df = yf.download(ticker, period="6mo", interval="1h", progress=False)
        data_mode = "Hourly · 6mo window"
    elif days_diff <= 365:
        # 6mo–1yr → fetch 1 year hourly (~1400 rows)
        df = yf.download(ticker, period="1y", interval="1h", progress=False)
        data_mode = "Hourly · 1yr window"
    else:
        # 1yr+ → daily data + today's live price
        df = yf.download(ticker, start=sd, end=today, progress=False)
        df_live = yf.download(ticker, period="1d", interval="1m", progress=False)
        if len(df_live) > 0:
            df_live.columns = [col[0] for col in df_live.columns]
            df_live = df_live[["Open","High","Low","Close","Volume"]]
            df_live.index = df_live.index.tz_localize(None)
            today_row = pd.DataFrame({
                "Open":   [float(df_live["Open"].iloc[0])],
                "High":   [float(df_live["High"].max())],
                "Low":    [float(df_live["Low"].min())],
                "Close":  [float(df_live["Close"].iloc[-1])],
                "Volume": [int(df_live["Volume"].sum())]
            }, index=[pd.Timestamp(today)])
            df = pd.concat([df, today_row])
            df = df[~df.index.duplicated(keep="last")]
        data_mode = "Daily"

    if len(df) == 0:
        return None

    df.columns = [col[0] for col in df.columns]
    df = df[["Open","High","Low","Close","Volume"]]
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.sort_index(inplace=True)

    # ── FEATURE ENGINEERING ──
    df["return"]   = df["Close"].pct_change() * 100
    df["lag1"]     = df["Close"].shift(1)
    df["lag2"]     = df["Close"].shift(2)
    df["lag3"]     = df["Close"].shift(3)
    df["ma8"]      = df["Close"].rolling(window=8).mean()
    df["ma15"]     = df["Close"].rolling(window=15).mean()
    df["ema21"]    = df["Close"].ewm(span=21, adjust=False).mean()
    df["volatile"] = df["return"].rolling(window=10).std()

    bb_mid         = df["Close"].rolling(window=20).mean()
    bb_std         = df["Close"].rolling(window=20).std()
    df["bb_upper"] = bb_mid + (2 * bb_std)
    df["bb_lower"] = bb_mid - (2 * bb_std)

    delta        = df["Close"].diff()
    gain         = delta.clip(lower=0).rolling(window=14).mean()
    loss         = (-delta.clip(upper=0)).rolling(window=14).mean()
    df["rsi14"]  = 100 - (100 / (1 + gain / loss))
    df["roc10"]  = df["Close"].pct_change(periods=10) * 100
    df["mom10"]  = df["Close"] - df["Close"].shift(10)

    highest_high  = df["High"].rolling(window=14).max()
    lowest_low    = df["Low"].rolling(window=14).min()
    df["stoch_k"] = ((df["Close"] - lowest_low) / (highest_high - lowest_low)) * 100

    ema12             = df["Close"].ewm(span=12, adjust=False).mean()
    ema26             = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(window=14).mean()

    df["vol_ma10"]  = df["Volume"].rolling(window=10).mean()
    df["vol_ratio"] = df["Volume"] / df["vol_ma10"]
    df["hl_range"]  = df["High"] - df["Low"]

    df.dropna(inplace=True)
    n_rows = len(df)

    if n_rows < 30:
        return None

    # ── ADAPTIVE FEATURE SELECTION ──
    # Rule: max features = rows / 10
    if n_rows >= 400:
        feature_cols = [
            "Close", "return", "lag1", "lag2", "lag3",
            "ma8", "ma15", "ema21",
            "volatile", "bb_upper", "bb_lower",
            "rsi14", "roc10", "mom10", "stoch_k",
            "macd", "macd_signal", "atr",
            "vol_ratio", "cluster"
        ]
        mode_label = f"Full · 20 features · {n_rows} rows"
    elif n_rows >= 150:
        feature_cols = [
            "Close", "return", "lag1", "lag2",
            "ma8", "ma15",
            "volatile", "bb_upper",
            "rsi14", "macd", "atr",
            "cluster"
        ]
        mode_label = f"Medium · 12 features · {n_rows} rows"
    else:
        feature_cols = [
            "Close", "return", "lag1",
            "ma8", "rsi14", "atr",
            "cluster"
        ]
        mode_label = f"Minimal · 7 features · {n_rows} rows"

    # ── CLUSTERING ──
    cluster_features = ["ma8", "return", "volatile", "rsi14"]
    scaler   = StandardScaler()
    x_scaled = scaler.fit_transform(df[cluster_features])
    kmeans   = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(x_scaled)

    # ── MODEL ──
    df["target"] = df["Close"].shift(-1)
    df.dropna(inplace=True)

    X = df[feature_cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    feature_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled  = feature_scaler.transform(X_test)

    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    train_preds = model.predict(X_train_scaled)

    rmse     = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae      = float(mean_absolute_error(y_test, predictions))
    r2       = float(r2_score(y_test, predictions))
    train_r2 = float(r2_score(y_train, train_preds))

    last_price  = float(df["Close"].iloc[-1])
    next_price  = float(model.predict(feature_scaler.transform(X.iloc[[-1]]))[0])
    change      = next_price - last_price
    change_pct  = (change / last_price) * 100
    current_vol = float(df["volatile"].iloc[-1])
    avg_vol     = float(df["volatile"].mean())

    return {
        "df":             df,
        "feature_cols":   feature_cols,
        "model":          model,
        "feature_scaler": feature_scaler,
        "X_train":        X_train,
        "X_test":         X_test,
        "y_train":        y_train,
        "y_test":         y_test,
        "predictions":    predictions,
        "train_preds":    train_preds,
        "rmse":           rmse,
        "mae":            mae,
        "r2":             r2,
        "train_r2":       train_r2,
        "last_price":     last_price,
        "next_price":     next_price,
        "change":         change,
        "change_pct":     change_pct,
        "current_vol":    current_vol,
        "avg_vol":        avg_vol,
        "split":          len(X_train),
        "data_mode":      data_mode,
        "mode_label":     mode_label,
        "n_rows":         n_rows,
    }

# =========================================================
# CHART DATA LOADER
# =========================================================

@st.cache_data(ttl=60)
def load_chart_data(ticker, period, interval):
    try:
        d = yf.download(ticker, period=period, interval=interval, progress=False)
        if len(d) == 0:
            return pd.DataFrame()
        d.columns = [col[0] for col in d.columns]
        d = d[["Open","High","Low","Close","Volume"]]
        d.index = pd.to_datetime(d.index)
        if d.index.tz is not None:
            d.index = d.index.tz_localize(None)
        return d
    except:
        return pd.DataFrame()

# =========================================================
# RUN MODEL
# =========================================================

with st.spinner("Fetching data and running model…"):
    R = build_model(ticker_name, start_date, model_alpha, n_clusters)

if R is None:
    st.error("❌ Not enough data to train. Try a different ticker or date range.")
    st.stop()

# unpack
df             = R["df"]
feature_cols   = R["feature_cols"]
model          = R["model"]
feature_scaler = R["feature_scaler"]
y_test         = R["y_test"]
y_train        = R["y_train"]
predictions    = R["predictions"]
train_preds    = R["train_preds"]
rmse           = R["rmse"]
mae            = R["mae"]
r2             = R["r2"]
train_r2       = R["train_r2"]
last_price     = R["last_price"]
next_price     = R["next_price"]
change         = R["change"]
change_pct     = R["change_pct"]
current_vol    = R["current_vol"]
avg_vol        = R["avg_vol"]
split          = R["split"]
X_test         = R["X_test"]
data_mode      = R["data_mode"]
mode_label     = R["mode_label"]
n_rows         = R["n_rows"]
overfit        = train_r2 - r2
overfit_lbl    = "Healthy ✓" if overfit <= 0.05 else "Overfitting ⚠"

# =========================================================
# SIGNAL
# =========================================================

if change_pct > 1.5:
    signal = "STRONG BUY"; signal_icon = "🚀"; signal_css = "strong-buy"
    signal_desc = f"Model expects a strong upward move · {change_pct:+.2f}%"
elif change_pct > 0.5:
    signal = "BUY";        signal_icon = "✅"; signal_css = "buy"
    signal_desc = f"Positive momentum detected · {change_pct:+.2f}% expected"
elif change_pct >= -0.5:
    signal = "HOLD";       signal_icon = "⏳"; signal_css = "hold"
    signal_desc = f"Sideways price action anticipated · {change_pct:+.2f}%"
else:
    signal = "AVOID";      signal_icon = "🚫"; signal_css = "avoid"
    signal_desc = f"Downside risk flagged · {change_pct:+.2f}% expected"

if current_vol > avg_vol * 1.5:
    risk_level = "HIGH RISK ⚠️";     risk_color = "#F87171"
    risk_note  = "Market unusually volatile. Even Buy signals carry extra risk."
elif current_vol > avg_vol:
    risk_level = "MODERATE RISK 🟠"; risk_color = "#FBBF24"
    risk_note  = "Volatility slightly above average. Proceed with caution."
else:
    risk_level = "LOW RISK 🟢";      risk_color = "#48BB78"
    risk_note  = "Market is calm. Signal is more reliable."

if "BUY" in signal and "LOW" in risk_level:
    verdict = "✅ Best case — strong opportunity in a stable market."
elif "BUY" in signal and "HIGH" in risk_level:
    verdict = "⚡ Opportunity exists but market is wild. Size down."
elif "BUY" in signal and "MODERATE" in risk_level:
    verdict = "🟡 Moderate opportunity with slightly elevated risk."
elif "HOLD" in signal and "LOW" in risk_level:
    verdict = "👀 Market stable but no strong move. Watch closely."
elif "HOLD" in signal and "HIGH" in risk_level:
    verdict = "🛑 No clear opportunity + volatile market. Stay out."
elif "AVOID" in signal:
    verdict = "🚫 Price drop expected. Do not enter."
else:
    verdict = "🔍 Monitor closely before deciding."

# =========================================================
# PLOTLY THEME
# =========================================================

DARK = dict(
    template="plotly_dark",
    paper_bgcolor="#0B0F1E",
    plot_bgcolor="#080C16",
    font=dict(family="Outfit, sans-serif", color="#6B7A94"),
    xaxis=dict(gridcolor="#141C2E", showgrid=True),
    yaxis=dict(gridcolor="#141C2E", showgrid=True),
    margin=dict(l=10, r=10, t=40, b=10)
)
CLUSTER_COLORS = ["#63B3ED", "#48BB78", "#FBBF24", "#F87171", "#9A66FF", "#FB923C"]

# =========================================================
# HERO
# =========================================================

st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-eyebrow">⚡ QuantAI · Machine Learning Stock Intelligence</div>
    <div class="hero-title">AI <span>Stock Prediction</span> Dashboard</div>
    <div class="hero-badges">
        <span class="hero-badge">{ticker_name}</span>
        <span class="hero-badge green">Ridge Regression</span>
        <span class="hero-badge purple">KMeans · {n_clusters} Clusters</span>
        <span class="hero-badge gold">{mode_label}</span>
        <span class="hero-badge">Data: {data_mode}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# KPI CARDS
# =========================================================

change_cls   = "up" if change >= 0 else "down"
change_arrow = "▲" if change >= 0 else "▼"

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-accent blue"></div>
        <div class="kpi-label">Current Price</div>
        <div class="kpi-val">₹{last_price:,.2f}</div>
        <div class="kpi-sub">Last known close</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-accent purple"></div>
        <div class="kpi-label">Predicted Next</div>
        <div class="kpi-val purple">₹{next_price:,.2f}</div>
        <div class="kpi-sub {change_cls}">{change_arrow} ₹{abs(change):.2f} · {change_pct:+.2f}%</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-accent {'green' if change >= 0 else 'red'}"></div>
        <div class="kpi-label">Expected Move</div>
        <div class="kpi-val {change_cls}">{change_arrow} {abs(change_pct):.2f}%</div>
        <div class="kpi-sub">Next session</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-accent blue"></div>
        <div class="kpi-label">Test R²</div>
        <div class="kpi-val blue">{r2:.4f}</div>
        <div class="kpi-sub">Train R² · {train_r2:.4f}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-accent {'green' if overfit <= 0.05 else 'red'}"></div>
        <div class="kpi-label">Overfit Gap</div>
        <div class="kpi-val {'up' if overfit <= 0.05 else 'down'}">{overfit:.4f}</div>
        <div class="kpi-sub">{overfit_lbl}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIGNAL + RISK + VERDICT
# =========================================================

st.markdown(f"""
<div class="signal-row">
    <div class="signal-card {signal_css}">
        <div class="signal-icon">{signal_icon}</div>
        <div>
            <div class="signal-lbl">Trading Signal</div>
            <div class="signal-name {signal_css}">{signal}</div>
            <div class="signal-desc">{signal_desc}</div>
        </div>
    </div>
    <div class="risk-card">
        <div class="signal-lbl">Risk Assessment</div>
        <div style="font-family:'Outfit',sans-serif;font-size:1.3rem;font-weight:700;color:{risk_color};margin:0.2rem 0;">{risk_level}</div>
        <div style="font-size:0.78rem;color:#4B5A72;margin-top:0.3rem;">{risk_note}</div>
        <div style="margin-top:0.7rem;font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#2A4A7A;">
            Current: {current_vol:.4f} &nbsp;|&nbsp; Avg: {avg_vol:.4f}
        </div>
    </div>
    <div class="verdict-card">
        <div class="signal-lbl">Combined Verdict</div>
        <div style="font-family:'Outfit',sans-serif;font-size:1rem;font-weight:600;color:#C8D0E0;margin-top:0.4rem;line-height:1.5;">{verdict}</div>
        <div style="margin-top:0.8rem;font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#2A4A7A;">
            RMSE ₹{rmse:.2f} &nbsp;·&nbsp; MAE ₹{mae:.2f} &nbsp;·&nbsp; α={model_alpha}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# STATS PILLS
# =========================================================

st.markdown(f"""
<div class="pill-row">
    <div class="pill"><span>RMSE</span><strong>₹{rmse:.2f}</strong></div>
    <div class="pill"><span>MAE</span><strong>₹{mae:.2f}</strong></div>
    <div class="pill"><span>Train R²</span><strong>{train_r2:.4f}</strong></div>
    <div class="pill"><span>Test R²</span><strong>{r2:.4f}</strong></div>
    <div class="pill"><span>Train Samples</span><strong>{split}</strong></div>
    <div class="pill"><span>Test Samples</span><strong>{len(X_test)}</strong></div>
    <div class="pill"><span>Features</span><strong>{len(feature_cols)}</strong></div>
    <div class="pill"><span>Rows</span><strong>{n_rows}</strong></div>
    <div class="pill"><span>Data Mode</span><strong>{data_mode}</strong></div>
    <div class="pill"><span>Clusters</span><strong>{n_clusters}</strong></div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Candlestick",
    "🎯  Prediction",
    "🧠  Features",
    "🧩  Clusters"
])

# ─────────────────────────────────────────────────────────
# TAB 1 — CANDLESTICK
# ─────────────────────────────────────────────────────────

presets = {
    "1D":  ("1d",  "30m"),
    "5D":  ("5d",  "60m"),
    "1M":  ("1mo", "360m"),
    "3M":  ("3mo", "1d"),
    "6M":  ("6mo", "1d"),
    "1Y":  ("1y",  "1d"),
    "MAX": ("max", "1d"),
}

with tab1:
    st.markdown('<div class="sec-hdr">Price Action · Moving Averages · Volume · RSI</div>', unsafe_allow_html=True)

    if "chart_range" not in st.session_state:
        st.session_state.chart_range = "3M"

    btn_cols = st.columns(len(presets))
    for i, (label, _) in enumerate(presets.items()):
        with btn_cols[i]:
            active = "🔵 " if st.session_state.chart_range == label else ""
            if st.button(f"{active}{label}", key=f"btn_{label}"):
                st.session_state.chart_range = label

    selected             = st.session_state.chart_range
    period_str, interval_str = presets[selected]

    with st.spinner(f"Loading {selected} chart…"):
        df_chart_raw = load_chart_data(ticker_name, period_str, interval_str)

    if df_chart_raw.empty:
        st.warning("No chart data available for this range. Try another timeframe.")
        df_chart = df.tail(90).copy()
    else:
        df_chart = df_chart_raw.copy()

    # indicators on chart data
    df_chart["ma8"]   = df_chart["Close"].rolling(8).mean()
    df_chart["ma15"]  = df_chart["Close"].rolling(15).mean()
    df_chart["ema21"] = df_chart["Close"].ewm(span=21, adjust=False).mean()
    _bb_m = df_chart["Close"].rolling(20).mean()
    _bb_s = df_chart["Close"].rolling(20).std()
    df_chart["bb_upper"] = _bb_m + 2 * _bb_s
    df_chart["bb_lower"] = _bb_m - 2 * _bb_s
    _d = df_chart["Close"].diff()
    _g = _d.clip(lower=0).rolling(14).mean()
    _l = (-_d.clip(upper=0)).rolling(14).mean()
    df_chart["rsi14"] = 100 - (100 / (1 + _g / _l))
    _e12 = df_chart["Close"].ewm(span=12, adjust=False).mean()
    _e26 = df_chart["Close"].ewm(span=26, adjust=False).mean()
    df_chart["macd"]        = _e12 - _e26
    df_chart["macd_signal"] = df_chart["macd"].ewm(span=9, adjust=False).mean()

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1: show_bb         = st.checkbox("Bollinger Bands", value=True)
    with c2: show_ema        = st.checkbox("EMA 21",          value=True)
    with c3: show_macd_panel = st.checkbox("MACD Panel",      value=False)

    rows    = 4 if show_macd_panel else 3
    heights = [0.55, 0.18, 0.14, 0.13] if show_macd_panel else [0.60, 0.22, 0.18]

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.025, row_heights=heights)

    fig.add_trace(go.Candlestick(
        x=df_chart.index, open=df_chart["Open"], high=df_chart["High"],
        low=df_chart["Low"], close=df_chart["Close"], name="Price",
        increasing_line_color="#48BB78", decreasing_line_color="#F87171",
        increasing_fillcolor="#48BB78",  decreasing_fillcolor="#F87171",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["ma8"],
        name="MA 8",  line=dict(color="#63B3ED", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["ma15"],
        name="MA 15", line=dict(color="#9A66FF", width=1.5)), row=1, col=1)

    if show_ema:
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["ema21"],
            name="EMA 21", line=dict(color="#FBBF24", width=1.2, dash="dot")), row=1, col=1)

    if show_bb:
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["bb_upper"],
            name="BB Upper", line=dict(color="#4B5A72", width=0.8, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["bb_lower"],
            name="BB Lower", line=dict(color="#4B5A72", width=0.8, dash="dash"),
            fill="tonexty", fillcolor="rgba(75,90,114,0.06)"), row=1, col=1)

    vol_colors = ["#48BB78" if c >= o else "#F87171"
                  for c, o in zip(df_chart["Close"], df_chart["Open"])]
    fig.add_trace(go.Bar(x=df_chart.index, y=df_chart["Volume"],
        name="Volume", marker_color=vol_colors, opacity=0.7), row=2, col=1)

    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["rsi14"],
        name="RSI 14", line=dict(color="#63B3ED", width=1.5)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#F87171", opacity=0.4, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#48BB78", opacity=0.4, row=3, col=1)

    if show_macd_panel:
        macd_hist_vals = df_chart["macd"] - df_chart["macd_signal"]
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["macd"],
            name="MACD", line=dict(color="#63B3ED", width=1.5)), row=4, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["macd_signal"],
            name="Signal", line=dict(color="#FBBF24", width=1.2)), row=4, col=1)
        fig.add_trace(go.Bar(x=df_chart.index, y=macd_hist_vals, name="Histogram",
            marker_color=["#48BB78" if v >= 0 else "#F87171" for v in macd_hist_vals],
            opacity=0.6), row=4, col=1)

    fig.update_layout(**DARK, height=820, showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=11)),
        xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    interval_labels = {"30m":"30-min candles","1h":"1-hour candles","90m":"90-min candles","1d":"Daily candles"}
    st.markdown(f"<p style='font-family:IBM Plex Mono,monospace;font-size:0.62rem;color:#2A4A7A;margin-top:-0.5rem;'>{selected} view · {interval_labels.get(interval_str,interval_str)} · {len(df_chart)} bars</p>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# TAB 2 — PREDICTION
# ─────────────────────────────────────────────────────────

with tab2:
    st.markdown('<div class="sec-hdr">Ridge Regression · Actual vs Predicted · Residual Analysis</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 1])

    with col_a:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=y_test.index, y=y_test.values, mode="lines", name="Actual",
            line=dict(color="#63B3ED", width=2)))
        fig2.add_trace(go.Scatter(x=y_test.index, y=predictions, mode="lines", name="Predicted",
            line=dict(color="#FBBF24", width=2, dash="dot")))
        fig2.update_layout(**DARK, height=420,
            title=dict(text="Actual vs Predicted Close Price (Test Set)", font=dict(size=14, color="#6B7A94")),
            legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig2, use_container_width=True)

        residuals = y_test.values - predictions
        fig_res = go.Figure()
        fig_res.add_trace(go.Bar(x=y_test.index, y=residuals,
            marker_color=["#48BB78" if r >= 0 else "#F87171" for r in residuals],
            opacity=0.75, name="Residual"))
        fig_res.add_hline(y=0, line_color="#4B5A72", line_dash="dash")
        fig_res.update_layout(**DARK, height=220,
            title=dict(text="Residuals (Actual − Predicted)", font=dict(size=13, color="#6B7A94")),
            showlegend=False)
        st.plotly_chart(fig_res, use_container_width=True)

    with col_b:
        st.markdown(f"""
        <div style="background:#0B0F1E;border:1px solid #141C2E;border-radius:16px;padding:1.4rem;">
            <div class="kpi-label">TEST R²</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:1.9rem;font-weight:700;color:#63B3ED;margin-bottom:1rem;">{r2:.4f}</div>
            <div class="kpi-label">TRAIN R²</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:1.9rem;font-weight:700;color:#9A66FF;margin-bottom:1rem;">{train_r2:.4f}</div>
            <div class="kpi-label">OVERFIT GAP</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:1.2rem;font-weight:700;color:{'#48BB78' if overfit<=0.05 else '#F87171'};margin-bottom:1rem;">{overfit:.4f}<br>{overfit_lbl}</div>
            <div class="kpi-label">RMSE</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:1.2rem;font-weight:700;color:#C8D0E0;margin-bottom:1rem;">₹{rmse:.2f}</div>
            <div class="kpi-label">MAE</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:1.2rem;font-weight:700;color:#C8D0E0;margin-bottom:1rem;">₹{mae:.2f}</div>
            <div class="kpi-label">TRAIN / TEST</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.85rem;color:#4B5A72;margin-bottom:0.5rem;">{split} / {len(y_test)}</div>
            <div class="kpi-label">DATA MODE</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:#4B5A72;">{data_mode}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# TAB 3 — FEATURE ANALYSIS
# ─────────────────────────────────────────────────────────

with tab3:
    st.markdown('<div class="sec-hdr">Feature Dataset · Ridge Coefficients · Correlation Heatmap</div>', unsafe_allow_html=True)

    col_f1, col_f2 = st.columns([1, 1])

    with col_f1:
        st.markdown("<p style='font-size:0.8rem;color:#4B5A72;'>Recent Feature Values — last 30 bars</p>", unsafe_allow_html=True)
        st.dataframe(df[feature_cols].tail(30).style.format("{:.3f}"),
                     use_container_width=True, height=380)

    with col_f2:
        coef_df = pd.DataFrame({
            "Feature":    feature_cols,
            "Importance": np.abs(model.coef_)
        }).sort_values("Importance", ascending=True)

        fig_imp = go.Figure(go.Bar(
            x=coef_df["Importance"], y=coef_df["Feature"], orientation="h",
            marker=dict(color=coef_df["Importance"],
                        colorscale=[[0,"#0F1E3A"],[0.5,"#2563EB"],[1,"#63B3ED"]])
        ))
        fig_imp.update_layout(**DARK, height=380,
            title=dict(text="Feature Importance  |Ridge Coefficients|", font=dict(size=13, color="#6B7A94")),
            showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)

    corr = df[feature_cols].corr()
    heat = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
        colorscale="RdBu", zmid=0,
        text=corr.round(2).values, texttemplate="%{text}", textfont=dict(size=8)
    ))
    heat.update_layout(**DARK, height=640,
        title=dict(text="Feature Correlation Matrix", font=dict(size=14, color="#6B7A94")))
    st.plotly_chart(heat, use_container_width=True)

# ─────────────────────────────────────────────────────────
# TAB 4 — CLUSTERS
# ─────────────────────────────────────────────────────────

with tab4:
    st.markdown('<div class="sec-hdr">KMeans Market Regime Clustering · Cluster Statistics</div>', unsafe_allow_html=True)

    col_c1, col_c2 = st.columns([3, 1])

    with col_c1:
        fig3 = go.Figure()
        for c in range(n_clusters):
            mask = df["cluster"] == c
            fig3.add_trace(go.Scatter(x=df.index[mask], y=df["Close"][mask],
                mode="markers", name=f"Cluster {c}",
                marker=dict(color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)], size=5, opacity=0.8)))
        fig3.update_layout(**DARK, height=480,
            title=dict(text="Price Coloured by Market Regime", font=dict(size=14, color="#6B7A94")),
            legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = go.Figure()
        for c in range(n_clusters):
            mask = df["cluster"] == c
            fig4.add_trace(go.Scatter(x=df["rsi14"][mask], y=df["volatile"][mask],
                mode="markers", name=f"Cluster {c}",
                marker=dict(color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)], size=5, opacity=0.7)))
        fig4.update_layout(**DARK, height=320,
            title=dict(text="RSI vs Volatility by Cluster", font=dict(size=13, color="#6B7A94")),
            xaxis_title="RSI (14)", yaxis_title="Volatility",
            legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig4, use_container_width=True)

    with col_c2:
        cluster_stats = df.groupby("cluster").agg({
            "Close":"mean","return":"mean","rsi14":"mean","volatile":"mean","atr":"mean"
        }).round(2)

        for idx, row in cluster_stats.iterrows():
            color     = CLUSTER_COLORS[int(idx) % len(CLUSTER_COLORS)]
            ret_color = "#48BB78" if row["return"] >= 0 else "#F87171"
            st.markdown(f"""
            <div style="background:#0B0F1E;border:1px solid {color}33;border-left:3px solid {color};
                        border-radius:12px;padding:1rem;margin-bottom:0.7rem;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:{color};letter-spacing:0.1em;margin-bottom:0.5rem;">CLUSTER {idx}</div>
                <div style="font-size:0.78rem;color:#6B7A94;line-height:2.2;">
                    <span style="color:#C8D0E0;">Avg Close</span> &nbsp;·&nbsp; ₹{row["Close"]:,.1f}<br>
                    <span style="color:{ret_color};">Avg Return</span> &nbsp;·&nbsp; {row["return"]:+.3f}%<br>
                    <span style="color:#C8D0E0;">Avg RSI</span> &nbsp;·&nbsp; {row["rsi14"]:.1f}<br>
                    <span style="color:#C8D0E0;">Volatility</span> &nbsp;·&nbsp; {row["volatile"]:.3f}<br>
                    <span style="color:#C8D0E0;">ATR</span> &nbsp;·&nbsp; ₹{row["atr"]:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        count_df = df["cluster"].value_counts().sort_index()
        fig_pie  = go.Figure(go.Pie(
            labels=[f"Cluster {i}" for i in count_df.index],
            values=count_df.values, hole=0.55,
            marker=dict(colors=CLUSTER_COLORS[:n_clusters])
        ))
        fig_pie.update_layout(
            template="plotly_dark", paper_bgcolor="#0B0F1E", plot_bgcolor="#0B0F1E",
            font=dict(family="Outfit, sans-serif", color="#6B7A94"),
            height=240, showlegend=True,
            title=dict(text="Regime Distribution", font=dict(size=12, color="#6B7A94")),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_pie, use_container_width=True)