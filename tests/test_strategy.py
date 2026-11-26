"""Tests for the risk-targeting strategy."""
import numpy as np
import pandas as pd

from signaldesk.returns import log_returns
from signaldesk.strategy import exposure_from_vol, forecast_vol_annual, strategy_returns


def _prices(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.012, n))), index=idx)


def _run(p, target=0.15, cap=1.0):
    e = exposure_from_vol(forecast_vol_annual(log_returns(p)), target, cap)
    return e, strategy_returns(p.pct_change(), e)


def test_no_look_ahead_in_exposure_and_pnl():
    # Change every price AFTER day t. Exposures up to t and P&L up to t must not move.
    p = _prices()
    t = 200
    p2 = p.copy()
    p2.iloc[t + 1:] *= np.linspace(0.6, 2.5, len(p) - t - 1)
    e1, s1 = _run(p)
    e2, s2 = _run(p2)
    cut = p.index[t]
    pd.testing.assert_series_equal(e1.loc[:cut], e2.loc[:cut])
    pd.testing.assert_series_equal(s1.loc[:cut], s2.loc[:cut])
    assert not np.allclose(e1.iloc[t + 2:], e2.iloc[t + 2:])   # the change does show up later


def test_exposure_is_capped_and_never_negative():
    vol = pd.Series([0.05, 0.10, 0.15, 0.30, 0.60])
    e = exposure_from_vol(vol, target_vol=0.15, max_exposure=1.5)
    assert np.allclose(e.values, [1.5, 1.5, 1.0, 0.5, 0.25])
    assert (e >= 0).all()
