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


# ---------------------------------------------------------------------------
# Hand-checkable examples: small enough to verify with pencil and paper.
# ---------------------------------------------------------------------------

def test_ewma_by_hand():
    # r = [1, 2, 3, 0]  ->  r^2 = [1, 4, 9, 0];  lambda = 0.5;  seed from first 2 days.
    #   h[0] = NaN                               (warm-up: not enough data yet)
    #   h[1] = mean(1, 4)              = 2.5     (initial state)
    #   h[2] = 0.5*2.5  + 0.5*9        = 5.75
    #   h[3] = 0.5*5.75 + 0.5*0        = 2.875
    r = pd.Series([1.0, 2.0, 3.0, 0.0])
    h = ewma_variance(r, lam=0.5, init_window=2)
    assert np.isnan(h.iloc[0])
    assert np.allclose(h.iloc[1:].values, [2.5, 5.75, 2.875])


def test_boxcar_by_hand():
    # Same r^2 = [1, 4, 9, 0], window N = 2:  [NaN, (1+4)/2, (4+9)/2, (9+0)/2]
    r = pd.Series([1.0, 2.0, 3.0, 0.0])
    h = rolling_variance(r, 2)
    assert np.isnan(h.iloc[0])
    assert np.allclose(h.iloc[1:].values, [2.5, 6.5, 4.5])


def test_ewma_impulse_response_and_half_life():
    # Feed a single spike r^2 = 1 into a filter at rest (all-zero input).
    # The output should be the impulse response (1 - lam) * lam^k, and it
    # should fall to half its peak after ln(0.5)/ln(lam) days (~11.2 for 0.94).
    lam = 0.94
    r = pd.Series(np.zeros(100))
    r.iloc[20] = 1.0                       # first 20 days are zero -> state starts at 0
    h = ewma_variance(r, lam=lam, init_window=20).iloc[20:].to_numpy()
    k = np.arange(len(h))
    assert np.allclose(h, (1 - lam) * lam**k)
    half_life = np.log(0.5) / np.log(lam)
    assert h[int(np.floor(half_life))] > h[0] / 2 > h[int(np.ceil(half_life))]


def test_ewma_has_unity_dc_gain():
    # Constant input power -> output settles at exactly that power (no bias).
    r = pd.Series(np.full(500, 2.0))        # r^2 = 4 every day
    assert np.isclose(ewma_variance(r, 0.97).iloc[-1], 4.0)
