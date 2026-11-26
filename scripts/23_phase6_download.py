"""Phase 6, step 2a: download the pre-registered universe (yfinance, adjusted close).

Saves to data/raw_phase6/; the original six in data/raw/ are not touched.
Run:  PYTHONPATH=src .venv/bin/python scripts/23_phase6_download.py
"""
import pandas as pd

from signaldesk.data import download_prices, save_prices
from signaldesk.universe import END_DATE, RAW6_DIR, UNIVERSE

# yfinance's end date is exclusive, so ask for the day after END_DATE.
end_excl = (pd.Timestamp(END_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
for t in UNIVERSE:
    df = download_prices(t, start="2015-01-01", end=end_excl)
    save_prices(df, t, RAW6_DIR)
    print(f"{t:8s} {len(df):5d} rows  {df.index[0].date()} -> {df.index[-1].date()}")
