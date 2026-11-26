"""Phase 6 tests: no look-ahead on new assets, identical scored days, original six untouched."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from signaldesk.ewma_family import c1_forecast, walk_forward
from signaldesk.phase6_scoring import MODELS, losses_from_forecasts
from signaldesk.universe import RAW6_DIR, UNIVERSE, load_asset, us_trading_days
from signaldesk.vol_models import garch_forecast, refit_days

ROOT = Path(__file__).resolve().parents[1]
needs6 = pytest.mark.skipif(not (RAW6_DIR / "BTC-USD.csv").exists(),
                            reason="run scripts/23_phase6_download.py first")

# The six original tickers; their SHA-256 checksums live in data/CHECKSUMS.sha256.
ORIGINAL_SIX = ["AAPL", "AMZN", "JPM", "MSFT", "SPY", "XOM"]
needs_raw = pytest.mark.skipif(not (ROOT / "data" / "raw" / "SPY.csv").exists(),
                               reason="price data not downloaded (scripts/01_download_prices.py)")


def _checksums() -> dict:
    lines = (ROOT / "data" / "CHECKSUMS.sha256").read_text().splitlines()
    return {rel: digest for digest, rel in (l.split(maxsplit=1) for l in lines)}


# ---- Nothing from the original six changed --------------------------------

def test_checksum_file_covers_all_data():
    names = set(_checksums())
    assert {f"raw/{t}.csv" for t in ORIGINAL_SIX} <= names
    assert {f"raw_phase6/{t}.csv" for t in UNIVERSE} <= names


@needs_raw
def test_original_six_data_unchanged():
    # If this fails after a fresh download, Yahoo has revised history since the
    # results were computed (see scripts/00_verify_data.py).
    sums = _checksums()
    for t in ORIGINAL_SIX:
        data = (ROOT / "data" / "raw" / f"{t}.csv").read_bytes()
        assert hashlib.sha256(data).hexdigest() == sums[f"raw/{t}.csv"], t


@needs_raw
def test_original_six_results_reproduce():
    # Recompute SPY's Phase 5 final-test QLIKE for the plain EWMA and compare with
    # the committed Phase 5 table: same code + same data -> same number.
    from signaldesk.data import load_prices
    r = 100 * np.log(load_prices("SPY")["AdjClose"]).diff().dropna()
    h = c1_forecast(r).shift(1)
    y = r ** 2
    ok = h.notna() & (h.index >= "2024-01-01") & (h.index <= "2026-11-25")
    # Phase 5 scored on days where every Phase 5 model had a forecast; from 2024
    # on that is every day, so the plain-EWMA mean is directly comparable.
    q = (y[ok] / h[ok] + np.log(h[ok])).mean()
    p5 = pd.read_csv(ROOT / "figures" / "phase5_final_test.csv")
    ref = p5[(p5.ticker == "SPY") & (p5.model == "C1 EWMA 0.94")].QLIKE.iloc[0]
    assert np.isclose(q, ref, atol=1e-4)


def test_original_six_not_in_universe():
    assert not set(UNIVERSE) & set(ORIGINAL_SIX)


# ---- No look-ahead on a new asset (bitcoin, with weekend folding) ----------

@needs6
def test_no_look_ahead_on_bitcoin():
    p = load_asset("BTC-USD")
    t = 1000                                    # a day between re-fits
    assert t not in refit_days(len(p) - 1)
    p2 = p.copy()
    p2.iloc[t + 1:] *= np.linspace(0.5, 2.0, len(p) - t - 1)
    r1 = 100 * np.log(p).diff().dropna()
    r2 = 100 * np.log(p2).diff().dropna()
    cut = p.index[t]
    for f in [c1_forecast, lambda r: walk_forward("C3 asymmetric", r)[0],
              lambda r: garch_forecast(r, asymmetric=True)[0]]:
        a, b = f(r1), f(r2)
        pd.testing.assert_series_equal(a.loc[:cut], b.loc[:cut])
        assert not np.allclose(a.iloc[t + 2:], b.iloc[t + 2:])


@needs6
def test_bitcoin_on_us_calendar():
    # Bitcoin is sampled on US trading days only; weekend moves fold into the next day.
    p = load_asset("BTC-USD")
    assert p.index.isin(us_trading_days()).all()
    assert (p.index.day_of_week < 5).all()


# ---- All models scored on the same days --------------------------------------

def test_same_days_for_all_models():
    # A tiny forecast table where one model starts late: scoring must drop the
    # days where ANY model is missing, for every model alike.
    idx = pd.bdate_range("2023-12-25", periods=8)
    F = pd.DataFrame({m: 1.0 for m in MODELS}, index=idx)
    F.iloc[:3, MODELS.index("GJR-GARCH(1,1)-t")] = np.nan
    F["r"] = 1.0
    F["ticker"] = "X"
    L = losses_from_forecasts(F.rename_axis("date").reset_index(), "2023-01-01")
    days = L.groupby("model").date.apply(lambda s: tuple(s))
    assert days.nunique() == 1 and len(days.iloc[0]) == 4   # 8 rows - 3 missing - 1 shift


@needs6
def test_same_days_on_real_forecasts():
    path = ROOT / "figures" / "phase6_forecasts.csv"
    if not path.exists():
        pytest.skip("run scripts/25_phase6_run_models.py first")
    F = pd.read_csv(path, parse_dates=["date"])
    L = losses_from_forecasts(F[F.ticker.isin(["BAC", "GLD", "BTC-USD"])], "2018-01-01")
    for t, g in L.groupby("ticker"):
        sets = g.groupby("model").date.apply(lambda s: tuple(s))
        assert sets.nunique() == 1, t
