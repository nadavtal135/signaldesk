"""Data-quality tests: fail loudly if the saved CSVs are broken."""
import pytest

from signaldesk.data import RAW_DIR, TICKERS, load_prices

pytestmark = pytest.mark.skipif(not (RAW_DIR / "SPY.csv").exists(),
                                reason="run scripts/01_download_prices.py first")


@pytest.mark.parametrize("ticker", TICKERS)
def test_prices_clean(ticker):
    df = load_prices(ticker)
    assert not df.isna().any().any(), "dropped samples (NaN)"
    assert (df["AdjClose"] > 0).all(), "price must be positive (we take its log later)"
    assert df.index.is_monotonic_increasing and df.index.is_unique, "time axis must be ordered"


def test_all_tickers_share_sample_clock():
    # Same dates for every ticker -> we can compare channels sample-by-sample.
    ref = load_prices("SPY").index
    for t in TICKERS:
        assert load_prices(t).index.equals(ref), t
