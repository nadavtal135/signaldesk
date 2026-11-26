"""Step 2b: sanity-check the downloaded data before we trust it.

DSP view: before any spectral analysis, check the sample stream itself --
dropped samples (NaN), impossible values (price <= 0), irregular sample
spacing (gaps longer than a normal weekend/holiday), and that all channels
share the same sample clock (identical date index).
"""
import pandas as pd

from signaldesk.data import TICKERS, load_prices

frames = {t: load_prices(t) for t in TICKERS}

ref_index = frames["SPY"].index
for t, df in frames.items():
    n_nan = int(df.isna().sum().sum())
    n_nonpos = int((df[["Open", "High", "Low", "Close", "AdjClose"]] <= 0).sum().sum())
    same_clock = df.index.equals(ref_index)
    # Calendar-day spacing between consecutive samples. 1 = weekday, 3 = weekend,
    # 4 = long weekend. Anything bigger is a market closure or missing data.
    gaps = df.index.to_series().diff().dt.days
    big = gaps[gaps > 4]
    # Adjustment factor: how much dividends/splits changed the price history.
    adj_ratio = (df["AdjClose"] / df["Close"]).iloc[0]
    print(f"{t:5s} rows={len(df)} NaN={n_nan} nonpos={n_nonpos} same_clock={same_clock} "
          f"max_gap={int(gaps.max())}d gaps>4d={len(big)} first AdjClose/Close={adj_ratio:.3f}")
    for d, g in big.items():
        print(f"       gap of {int(g)} days ending {d.date()}")

# Trading days per year: our "sample rate" in finance units.
per_year = ref_index.year.value_counts().sort_index()
print("\nSamples per calendar year:", dict(per_year))
