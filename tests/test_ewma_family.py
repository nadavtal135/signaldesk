"""Tests for the Phase 5 EWMA-family candidates."""
import numpy as np
import pandas as pd
import pytest

from signaldesk.ewma_family import (CANDIDATES, c1_forecast, c2_path, c3_path, c4_path,
                                    c5_path, ewma_filter, refit_days, walk_forward)
from signaldesk.returns import log_returns
from signaldesk.volatility import ewma_variance

FAST = {"first_fit": 200, "every": 50}


def _prices(n=420, seed=0):
    """Synthetic prices with volatility clustering and a down-day asymmetry."""
    rng = np.random.default_rng(seed)
    h, r = 1.0, np.zeros(n)
    for t in range(n):
        r[t] = np.sqrt(h) * rng.standard_t(6) / np.sqrt(1.5)
        h = 0.05 + (0.06 + 0.08 * (r[t] < 0)) * r[t] ** 2 + 0.85 * h
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(r / 100)), index=idx)


# ---- (a) look-ahead, including across a re-fit ----------------------------

@pytest.mark.parametrize("name", list(CANDIDATES))
def test_no_look_ahead(name):
    p = _prices()
    t = 270                                   # between re-fits 249 and 299
    assert t not in refit_days(len(p), **FAST)
    p2 = p.copy()
    p2.iloc[t + 1:] *= np.linspace(0.7, 1.6, len(p) - t - 1)
    h1, _ = walk_forward(name, 100 * log_returns(p), **FAST)
    h2, _ = walk_forward(name, 100 * log_returns(p2), **FAST)
    cut = p.index[t]
    pd.testing.assert_series_equal(h1.loc[:cut], h2.loc[:cut])
    assert not np.allclose(h1.iloc[t + 2:], h2.iloc[t + 2:])


# ---- (b) hand-checkable examples ------------------------------------------

def test_two_speed_by_hand():
    # r^2 = [1, 4, 9, 0], seed from first 2 days (2.5).
    #   fast (lambda 0.5): 2.5, 0.5*2.5+0.5*9 = 5.75,  0.5*5.75 = 2.875
    #   slow (lambda 0.9): 2.5, 0.9*2.5+0.1*9 = 3.15,  0.9*3.15 = 2.835
    #   w = 0.5 mix:       2.5, 4.45,                   2.855
    r2 = np.array([1.0, 4.0, 9.0, 0.0])
    fast, slow = ewma_filter(r2, 0.5, 2), ewma_filter(r2, 0.9, 2)
    assert np.allclose(fast[1:], [2.5, 5.75, 2.875])
    assert np.allclose(slow[1:], [2.5, 3.15, 2.835])
    assert np.allclose(0.5 * fast[1:] + 0.5 * slow[1:], [2.5, 4.45, 2.855])


def test_asymmetric_by_hand():
    # r = [1, 2, -3, 0], a = 1: the down day's r^2 = 9 counts double -> 18.
    #   input x = [1, 4, 18, 0]; lambda 0.5, seed from 2 days = 2.5
    #   h = 2.5, 0.5*2.5 + 0.5*18 = 10.25, 0.5*10.25 = 5.125   (c = 1 here)
    from signaldesk.ewma_family import asym_input
    r = np.array([1.0, 2.0, -3.0, 0.0])
    x = asym_input(r, 1.0)
    assert np.allclose(x, [1, 4, 18, 0])
    assert np.allclose(ewma_filter(x, 0.5, 2)[1:], [2.5, 10.25, 5.125])


def test_level_correction_removes_bias():
    # With c applied, the average forecast level matches the average r^2.
    from signaldesk.ewma_family import level_correction
    r = np.random.default_rng(3).standard_normal(5000)
    c = level_correction(r, 1.0)
    assert 0.6 < c < 0.7                          # ~1/(1 + 0.5) for symmetric returns
    h = c3_path(r, 0.94, 1.0, c)
    assert np.isclose(np.nanmean(h), np.mean(r ** 2), rtol=0.05)


# ---- (c) positivity -------------------------------------------------------

@pytest.mark.parametrize("name", list(CANDIDATES))
def test_forecasts_positive_on_real_data(name):
    from signaldesk.data import RAW_DIR, load_prices
    if not (RAW_DIR / "SPY.csv").exists():
        pytest.skip("run scripts/01_download_prices.py first")
    r = 100 * log_returns(load_prices("SPY")["AdjClose"])
    h, _ = walk_forward(name, r)
    h = h.dropna()
    assert len(h) > 1000 and (h > 0).all()


# ---- (d) special cases reduce to the plain EWMA ----------------------------

def test_reductions_to_plain_ewma():
    r = np.random.default_rng(1).standard_normal(400)
    base = ewma_variance(pd.Series(r), 0.94).to_numpy()
    assert np.allclose(ewma_filter(r ** 2, 0.94), base, equal_nan=True)   # lfilter == loop
    assert np.allclose(c2_path(r, 0.94, 0.05, 1.0), base, equal_nan=True)  # w = 1
    assert np.allclose(c3_path(r, 0.94, 0.0, 1.0), base, equal_nan=True)   # a = 0
    assert np.allclose(c4_path(r, 0.94, 0.05, 1.0, 0.0, 1.0), base, equal_nan=True)
    assert np.allclose(c5_path(r, 0.94, 0.0), base, equal_nan=True)        # k = 0
    assert np.allclose(c1_forecast(pd.Series(r)).to_numpy(), base, equal_nan=True)
