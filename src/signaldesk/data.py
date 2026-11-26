"""Loading and saving daily price data.

DSP view: each ticker is a discretely-sampled signal. The sample clock is
"one trading day", which is NOT uniform in wall-clock time (weekends and
holidays are skipped). We treat trading days as the uniform sample index,
which is the standard convention.
"""
from pathlib import Path

import pandas as pd

TICKERS = ["SPY", "AAPL", "MSFT", "AMZN", "JPM", "XOM"]
START = "2015-01-01"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def download_prices(ticker: str, start: str = START, end: str | None = None) -> pd.DataFrame:
    """Download daily OHLCV + adjusted close for one ticker from Yahoo via yfinance.

    We ask for auto_adjust=False so we keep both price columns:
      - Close:    already corrected for stock splits by Yahoo (a split, e.g.
                  1 share -> 4 shares, would otherwise look like a -75% crash).
      - AdjClose: ALSO corrected for dividends (cash paid to shareholders,
                  which makes the price drop by that amount on payout day).
    In DSP terms, both corrections remove step discontinuities that come from
    bookkeeping, not from the market. We use AdjClose for analysis.
    """
    import yfinance as yf  # imported here so the rest of the module works offline

    df = yf.download(ticker, start=start, end=end, auto_adjust=False,
                     progress=False, multi_level_index=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker} (network blocked?)")
    df.index.name = "Date"
    df = df.rename(columns={"Adj Close": "AdjClose"})
    return df[["Open", "High", "Low", "Close", "AdjClose", "Volume"]]


def save_prices(df: pd.DataFrame, ticker: str, out_dir: Path = RAW_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticker}.csv"
    df.to_csv(path)
    return path


def load_prices(ticker: str, in_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read a saved CSV back, with dates parsed as the index."""
    return pd.read_csv(in_dir / f"{ticker}.csv", index_col="Date", parse_dates=True)
