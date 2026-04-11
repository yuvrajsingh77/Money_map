import yfinance as yf

ticker_name = "RELIANCE.NS"

df_live = yf.download(ticker_name, period="1d", interval="1m")
df_live.columns = [col[0] for col in df_live.columns]

print(df_live[['Open', 'High', 'Low', 'Close', 'Volume']].tail(1).to_string())