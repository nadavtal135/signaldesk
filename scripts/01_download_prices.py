"""Step 2: download daily prices for all tickers and save one CSV each.

Run:  .venv/bin/python scripts/01_download_prices.py
"""
from signaldesk.data import TICKERS, download_prices, save_prices

for t in TICKERS:
    df = download_prices(t)
    path = save_prices(df, t)
    print(f"{t}: {len(df)} rows, {df.index[0].date()} -> {df.index[-1].date()}  -> {path.name}")
