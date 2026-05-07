import yfinance as yf
import time
from datetime import datetime, date
import pandas as pd

tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "MARUTI.NS"]

# --- Part 1: historical data for training ---
print("HISTORICAL DATA (for training the model)")
historical_data = {}
for t in tickers:
    df = yf.download(t, start="2024-01-01", end=date.today())
    df.columns = [col[0] for col in df.columns]
    historical_data[t] = df
    print(f"  {t:<15} | {len(df)} days of data | Latest close: ₹{df['Close'].iloc[-1]:.2f}")
    
# --- Part 2: live prices ---
print("LIVE PRICES (current market)")

print(f"\n[{datetime.now().strftime('%H:%M:%S')}]")
for t in tickers:
    ticker = yf.Ticker(t)
    price  = ticker.fast_info['last_price']
    high   = ticker.fast_info['day_high']
    low    = ticker.fast_info['day_low']
    volume = ticker.fast_info['last_volume']
    print(f"  {t:<15} | Price: ₹{price:.2f} | High: ₹{high:.2f} | Low: ₹{low:.2f} | Volume: {volume:,}")
    

# --- Part 3: show sample historical data for one stock ---

print("SAMPLE HISTORICAL DATA — RELIANCE.NS")
print(historical_data["RELIANCE.NS"])
