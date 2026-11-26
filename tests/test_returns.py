import numpy as np
import pandas as pd

from signaldesk.returns import log_returns


def test_log_returns_values():
    p = pd.Series([100.0, 110.0, 99.0])
    r = log_returns(p)
    assert np.allclose(r.values, [np.log(1.1), np.log(0.9)])


def test_log_returns_are_causal():
    # Changing a FUTURE price must not change any PAST return (no look-ahead).
    p = pd.Series(np.linspace(100, 120, 50))
    r_before = log_returns(p)
    p_future_changed = p.copy()
    p_future_changed.iloc[-1] *= 2
    r_after = log_returns(p_future_changed)
    assert r_before.iloc[:-1].equals(r_after.iloc[:-1])
