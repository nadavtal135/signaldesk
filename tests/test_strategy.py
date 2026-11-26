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


# ---------------------------------------------------------------------------
# Hand-checkable example: exposure, turnover, cost and P&L over 4 days.
# ---------------------------------------------------------------------------

def test_exposure_cost_pnl_by_hand():
    from signaldesk.strategy import turnover
    # Inputs (target 20%/yr, cap 1.0, cost 0.05% of amount traded):
    #   forecast vol  = [20%, 40%, 10%, 20%]
    #   stock return  = [ -,  +2%, -4%, +1%]   (day 0's return is never used)
    # Exposure = min(1, 0.20 / vol):
    #   [0.20/0.20, 0.20/0.40, min(1, 0.20/0.10), 0.20/0.20] = [1.0, 0.5, 1.0, 1.0]
    # Amount traded |e[t] - e[t-1]| (start from 0 = no position):
    #   [1.0, 0.5, 0.5, 0.0]
    # Gross P&L = yesterday's exposure x today's return:
    #   day0: 0 (nothing held yet)   day1: 1.0 x +2% = +2%
    #   day2: 0.5 x -4% = -2%        day3: 1.0 x +1% = +1%
    # Cost = 0.0005 x traded = [0.0005, 0.00025, 0.00025, 0]
    # Net = gross - cost = [-0.0005, 0.01975, -0.02025, 0.01]
    vol = pd.Series([0.20, 0.40, 0.10, 0.20])
    R = pd.Series([np.nan, 0.02, -0.04, 0.01])
    e = exposure_from_vol(vol, target_vol=0.20, max_exposure=1.0)
    assert np.allclose(e.values, [1.0, 0.5, 1.0, 1.0])
    assert np.allclose(turnover(e).values, [1.0, 0.5, 0.5, 0.0])
    net = strategy_returns(R, e, cost_rate=0.0005)
    assert np.allclose(net.values, [-0.0005, 0.01975, -0.02025, 0.01])


def test_max_drawdown_by_hand():
    from signaldesk.backtest import metrics
    # Net returns +10%, -50%, +20%: account 1.10 -> 0.55 -> 0.66.
    # Peak 1.10, lowest point after it 0.55 -> drawdown = 0.55/1.10 - 1 = -50%.
    bt = pd.DataFrame({"net": [0.10, -0.50, 0.20], "exposure": 1.0, "trade": 0.0})
    m = metrics(bt)
    assert np.isclose(m["max_drawdown_%"], -50.0)
    assert np.isclose(m["worst_day_%"], -50.0)


def test_no_look_ahead_full_backtest_pipeline():
    # Same idea as the exposure test, through the full run_strategy pipeline.
    from signaldesk.backtest import run_strategy
    p = _prices()
    t = 200
    p2 = p.copy()
    p2.iloc[t + 1:] *= 1.5
    cut = p.index[t]
    a = run_strategy(p, 0.15, 1.0).loc[:cut]
    b = run_strategy(p2, 0.15, 1.0).loc[:cut]
    pd.testing.assert_frame_equal(a, b)


def test_existing_position_is_not_a_new_trade_after_slicing():
    # Regression test for a bug fixed in step 4: buy-and-hold must show zero
    # trading in a later period, and a single entry trade at the very start.
    from signaldesk.backtest import constant_strategy
    R = _prices().pct_change()
    bh = constant_strategy(R, 1.0)
    assert bh["trade"].iloc[0] == 1.0
    assert bh["trade"].iloc[100:].sum() == 0.0
