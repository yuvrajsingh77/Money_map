import yfinance as yf
import mplfinance as mpf
from datetime import date

# download data
df = yf.download("RELIANCE.NS", start="2025-01-01", end=date.today())
df.columns = [col[0] for col in df.columns]

# mplfinance needs exactly these 4 columns — Open, High, Low, Close
# and the index must be dates
mpf.plot(df,
         type='candle',       # candlestick chart
         style='charles',     # visual style
         title='RELIANCE.NS', # title
         ylabel='Price (₹)',  # y axis label
         volume=True,         # show volume bars at bottom
         figsize=(14, 8))