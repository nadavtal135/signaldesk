"""Phase 6 universe (fixed in notes/phase6_plan.md before any data was downloaded)."""
from pathlib import Path

import pandas as pd

from signaldesk.data import load_prices

RAW6_DIR = Path(__file__).resolve().parents[2] / "data" / "raw_phase6"
END_DATE = "2026-11-25"      # same last day as the original data

CLASSES = {
    "stock": ["JNJ", "PG", "KO", "WMT", "CVX", "BAC", "GS", "INTC", "PFE", "HD",
              "CAT", "UNH", "NVDA", "T", "MRK"],
    "equity ETF": ["QQQ", "IWM", "XLE", "XLF", "XLU"],
    "bonds": ["TLT"],
    "gold": ["GLD"],
    "international": ["EFA", "EEM"],
    "oil": ["USO"],
    "bitcoin": ["BTC-USD"],
}
UNIVERSE = [t for ts in CLASSES.values() for t in ts]
ASSET_CLASS = {t: c for c, ts in CLASSES.items() for t in ts}
EQUITY_TYPE = CLASSES["stock"] + CLASSES["equity ETF"] + CLASSES["international"]   # 22 for H2


def us_trading_days() -> pd.DatetimeIndex:
    """The US trading calendar = the days SPY traded (from the frozen original data)."""
    return load_prices("SPY").index


def load_asset(ticker: str) -> pd.Series:
    """Adjusted close on the US trading calendar.

    Bitcoin trades every day; we keep only its closes on US trading days, so
    weekend/holiday moves fold into the next US trading day's return (exactly
    like weekend news for a stock). Every asset then has one sample per US
    trading day.
    """
    p = load_prices(ticker, RAW6_DIR)["AdjClose"]
    if ticker == "BTC-USD":
        p = p.reindex(us_trading_days()).dropna()
    return p
