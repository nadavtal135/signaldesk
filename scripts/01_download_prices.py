"""Step 2: download daily prices for the original six tickers and save one CSV each.

The end date is pinned to the last day of the research data (2026-11-25), so
re-running this cannot silently add new days and shift every result.
Note: Yahoo revises adjusted prices over time (new dividends), so a fresh
download may still differ slightly from the CSVs used in the notes.

Run:  PYTHONPATH=src .venv/bin/python scripts/01_download_prices.py
"""
from signaldesk.data import TICKERS, download_prices, save_prices

END_DATE = "2026-11-25"
END_EXCLUSIVE = "2026-11-26"    # yfinance's end date is exclusive

for t in TICKERS:
    df = download_prices(t, end=END_EXCLUSIVE)
    path = save_prices(df, t)
    print(f"{t}: {len(df)} rows, {df.index[0].date()} -> {df.index[-1].date()}  -> {path.name}")
