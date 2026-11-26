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

# SHA-256 of the original six CSVs as committed in Phase 1 (commit 9d37924).
ORIGINAL_SIX = {
    "AAPL": "b456536833256e5dbba09bb09acfe60f8a7add98827cf5743ce5ff34addb61cc",
    "AMZN": "74fe88e67351851d472e739ec105931b02b32eec1b7a94ab918a700726df6e45",
    "JPM": "16eec21c680993116cd03f5b9e78350d0fdd2a7f9f8ede7780aa2a2a24eee08b",
    "MSFT": "f9e2e65387119410ecfeb30d96734142395f276825e9339fc359d637c91b65fa",
    "SPY": "95c84bb4f85fdc8e6df0aad819f57cd1ccc0c42d3f7f2eb5726b38c0b24041d4",
    "XOM": "cc537d66cca62ac442d9b726eefad6711ccb44ce93801ad8b64fc60881d69977",
}


# ---- Nothing from the original six changed --------------------------------

def test_original_six_data_unchanged():
    for t, digest in ORIGINAL_SIX.items():
        data = (ROOT / "data" / "raw" / f"{t}.csv").read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, t


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
