"""Tests for the volatility estimators. The look-ahead test is the important one."""
import numpy as np
import pandas as pd
import pytest

from signaldesk.returns import log_returns
from signaldesk.volatility import constant_variance, ewma_variance, rolling_variance

ESTIMATORS = {
    "rolling10": lambda r: rolling_variance(r, 10),
    "rolling60": lambda r: rolling_variance(r, 60),
    "ewma0.94": lambda r: ewma_variance(r, 0.94),
    "ewma0.97": lambda r: ewma_variance(r, 0.97),
}


def _prices(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)


@pytest.mark.parametrize("name", ESTIMATORS)
def test_no_look_ahead(name):
    # Wildly change every price AFTER day t. Every estimate up to day t must be identical.
    est = ESTIMATORS[name]
    p = _prices()
    t = 200
    p_changed = p.copy()
    p_changed.iloc[t + 1:] *= np.linspace(0.5, 3.0, len(p) - t - 1)
    h1 = est(log_returns(p))
    h2 = est(log_returns(p_changed))
    cutoff = p.index[t]
    pd.testing.assert_series_equal(h1.loc[:cutoff], h2.loc[:cutoff])
    # ...and the change must actually show up afterwards (the test can fail).
    assert not np.allclose(h1.loc[cutoff:].iloc[2:], h2.loc[cutoff:].iloc[2:])


def test_constant_baseline_ignores_test_period():
    # The baseline may only use training data: changing test-period prices must not move it.
    p = _prices()
    train_end = p.index[200]
    p_changed = p.copy()
    p_changed.iloc[201:] *= 2.0
    a = constant_variance(log_returns(p), train_end)
    b = constant_variance(log_returns(p_changed), train_end)
    assert a.iloc[0] == b.iloc[0]
