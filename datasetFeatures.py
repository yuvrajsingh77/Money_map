import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date

# download data
df = yf.download("RELIANCE.NS", start="2025-01-01", end=date.today())
df.columns = [col[0] for col in df.columns]
df['return'] = df['Close'].pct_change() * 100
df.dropna(inplace=True)

df = df.tail(20)

fig, axes = plt.subplots(4, 1, figsize=(14, 16))

# open
axes[0].bar(range(len(df)), df['Open'], color='steelblue')
axes[0].set_title('Open Price')
axes[0].set_ylabel('Price (₹)')
axes[0].set_xticks(range(len(df)))
axes[0].set_xticklabels([d.strftime('%d %b') for d in df.index], rotation=45)

# close
axes[1].bar(range(len(df)), df['Close'], color='purple')
axes[1].set_title('Close Price')
axes[1].set_ylabel('Price (₹)')
axes[1].set_xticks(range(len(df)))
axes[1].set_xticklabels([d.strftime('%d %b') for d in df.index], rotation=45)

# volume
axes[2].bar(range(len(df)), df['Volume'], color='orange')
axes[2].set_title('Volume')
axes[2].set_ylabel('Volume')
axes[2].set_xticks(range(len(df)))
axes[2].set_xticklabels([d.strftime('%d %b') for d in df.index], rotation=45)

# return
colors = ['green' if r >= 0 else 'red' for r in df['return']]
axes[3].bar(range(len(df)), df['return'], color=colors)
axes[3].axhline(y=0, color='black', linewidth=0.8, linestyle='--')
axes[3].set_title('Daily Return (%)')
axes[3].set_ylabel('Return (%)')
axes[3].set_xlabel('Date')
axes[3].set_xticks(range(len(df)))
axes[3].set_xticklabels([d.strftime('%d %b') for d in df.index], rotation=45)

plt.tight_layout()
plt.show()