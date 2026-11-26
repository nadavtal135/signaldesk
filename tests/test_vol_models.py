"""Tests for the Phase 4 forecasters: look-ahead, hand-checkable recursion, positivity."""
import numpy as np
import pandas as pd
import pytest

from signaldesk.returns import log_returns
from signaldesk.vol_models import garch_forecast, garch_recursion, har_forecast, refit_days
from signaldesk.volatility import ewma_variance

# Small schedule so the tests run fast: first fit after 200 days, re-fit every 50.
FAST = {"first_fit": 200, "every": 50}
MODELS = {
    "garch": lambda r: garch_forecast(r, asymmetric=False, **FAST)[0],
    "gjr": lambda r: garch_forecast(r, asymmetric=True, **FAST)[0],
    "har": lambda r: har_forecast(r, **FAST)[0],
}


def _garch_like_prices(n=420, seed=0):
    """Synthetic prices with volatility clustering (so the GARCH fits are sensible)."""
    rng = np.random.default_rng(seed)
    h, r = 1.0, np.zeros(n)
    for t in range(n):
        r[t] = np.sqrt(h) * rng.standard_t(6) / np.sqrt(1.5)
        h = 0.05 + 0.10 * r[t] ** 2 + 0.85 * h
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(r / 100)), index=idx)


# ---- (a) No look-ahead ----------------------------------------------------

@pytest.mark.parametrize("name", MODELS)
def test_no_look_ahead(name):
    # Day t = 270 sits between re-fits (199, 249, 299, ...). Changing every price
    # after t changes the 299 re-fit and later forecasts -- but nothing up to t.
    p = _garch_like_prices()
    t = 270
    assert t not in refit_days(len(p), **FAST)
    p2 = p.copy()
    p2.iloc[t + 1:] *= np.linspace(0.7, 1.6, len(p) - t - 1)
    h1 = MODELS[name](100 * log_returns(p))
    h2 = MODELS[name](100 * log_returns(p2))
    cut = p.index[t]
    pd.testing.assert_series_equal(h1.loc[:cut], h2.loc[:cut])
    assert not np.allclose(h1.iloc[t + 2:], h2.iloc[t + 2:])   # the test can fail


# ---- (b) Hand-checkable recursion -----------------------------------------

def test_garch_recursion_by_hand():
    # r = [1, 2, -1, 0], seed from the first 2 days: h[1] = mean(1, 4) = 2.5
    # GARCH  omega=0.1, alpha=0.1, beta=0.8:
    #   h[2] = 0.1 + 0.1*(-1)^2 + 0.8*2.5 = 2.2
    #   h[3] = 0.1 + 0.1*0      + 0.8*2.2 = 1.86
    r = np.array([1.0, 2.0, -1.0, 0.0])
    h = garch_recursion(r, omega=0.1, alpha=0.1, beta=0.8, init_window=2)
    assert np.isnan(h[0])
    assert np.allclose(h[1:], [2.5, 2.2, 1.86])


def test_gjr_recursion_by_hand():
    # Same, with gamma = 0.2: day 2 is a DOWN day, so its gain is 0.1 + 0.2 = 0.3:
    #   h[2] = 0.1 + 0.3*1 + 0.8*2.5 = 2.4
    #   h[3] = 0.1 + 0.1*0 + 0.8*2.4 = 2.02   (r = 0 is not a down day)
    r = np.array([1.0, 2.0, -1.0, 0.0])
    h = garch_recursion(r, omega=0.1, alpha=0.1, beta=0.8, gamma=0.2, init_window=2)
    assert np.allclose(h[1:], [2.5, 2.4, 2.02])


def test_ewma_is_garch_with_no_setpoint():
    # EWMA = GARCH with omega = 0, alpha = 1 - lambda, beta = lambda.
    r = pd.Series(np.random.default_rng(1).standard_normal(300))
    g = garch_recursion(r.to_numpy(), omega=0.0, alpha=0.06, beta=0.94)
    assert np.allclose(g, ewma_variance(r, 0.94).to_numpy(), equal_nan=True)


# ---- (c) Strictly positive forecasts --------------------------------------

def test_all_forecasts_positive_on_real_data():
    from signaldesk.data import RAW_DIR, load_prices
    if not (RAW_DIR / "SPY.csv").exists():
        pytest.skip("run scripts/01_download_prices.py first")
    r = 100 * log_returns(load_prices("SPY")["AdjClose"])
    for f in [lambda r: ewma_variance(r, 0.94), lambda r: garch_forecast(r)[0],
              lambda r: garch_forecast(r, asymmetric=True)[0], lambda r: har_forecast(r)[0]]:
        h = f(r).dropna()
        assert len(h) > 1000 and (h > 0).all()


def test_har_stays_positive_through_a_dead_signal():
    # A long stretch of exactly-zero returns makes every HAR input zero; the
    # forecast must still be > 0 (intercept >= 0 plus the safety floor).
    p = _garch_like_prices()
    r = 100 * log_returns(p)
    r.iloc[300:360] = 0.0
    h = har_forecast(r, **FAST)[0].dropna()
    assert (h > 0).all()
