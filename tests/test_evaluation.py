import numpy as np
import pandas as pd

from signaldesk.evaluation import align_forecast, qlike_loss


def test_align_pairs_forecast_with_next_day():
    idx = pd.bdate_range("2024-01-01", periods=4)
    r = pd.Series([0.1, 0.2, 0.3, 0.4], index=idx)
    h = pd.Series([10.0, 20.0, 30.0, 40.0], index=idx)
    df = align_forecast(r, h)
    # forecast made on day 0 (10.0) is scored against r[1]^2 = 0.04, on day 1's row
    assert df.index[0] == idx[1]
    assert df["h"].iloc[0] == 10.0 and np.isclose(df["y"].iloc[0], 0.04)
    assert len(df) == 3  # the last forecast has no "tomorrow" yet


def test_qlike_is_minimised_by_true_variance():
    # If r ~ N(0, s2), expected QLIKE is minimised at h = s2.
    rng = np.random.default_rng(0)
    y = rng.normal(0, 2.0, 200_000) ** 2
    scores = {h: qlike_loss(y, h).mean() for h in [2.0, 3.0, 4.0, 5.0, 6.0]}
    assert min(scores, key=scores.get) == 4.0
