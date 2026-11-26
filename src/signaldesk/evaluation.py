"""Scoring volatility forecasts.

Setup: h[t] is a variance forecast made at the close of day t for day t+1.
The "truth" we compare it with is the realised r[t+1]^2 -- itself a very noisy,
single-sample measurement of the true noise power (roughly chi-squared with
1 degree of freedom times the true power). A good score must not be fooled by
that measurement noise.

MSE   = mean( (r^2 - h)^2 )
  The familiar squared-error loss. Fair in expectation (the best forecast is
  still the true power), but its value is dominated by a handful of crash days:
  r^2 is heavy-tailed, and squaring it again makes one 10% day weigh as much
  as ~10,000 normal days. So MSE is a very noisy yardstick.

QLIKE = mean( r^2 / h + ln(h) )
  The negative log-likelihood of r ~ Gaussian(0, h), up to constants. In DSP
  terms it scores the *ratio* r^2/h rather than the difference, so errors are
  judged in relative terms (like dB): forecasting 1 when the truth is 2 costs
  about the same in a calm month as forecasting 10 vs 20 in a crash. It also
  punishes UNDER-forecasting risk more than over-forecasting it (r^2/h blows up
  as h -> 0), which is the costly direction in practice. And, like MSE, it
  ranks forecasts correctly even though r^2 is a noisy proxy (Patton, 2011).
  Lower is better; only differences between models are meaningful.
"""
import numpy as np
import pandas as pd


def mse_loss(y: np.ndarray, h: np.ndarray) -> np.ndarray:
    return (y - h) ** 2


def qlike_loss(y: np.ndarray, h: np.ndarray) -> np.ndarray:
    return y / h + np.log(h)


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, max_lag: int = 10):
    """Is model A's average loss really lower than model B's, or just luck?

    d[t] = loss_a - loss_b. We test mean(d) = 0. Because the d[t] are
    correlated in time (volatility clusters), the standard error uses a
    Newey-West estimate: the variance of the mean of a coloured process is
    2*pi*PSD_d(0)/n, not var(d)/n -- the low-frequency power matters.
    Returns (t_stat, mean_d). t < -2 ~ A significantly better at ~95%.
    """
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    dc = d - d.mean()
    lrv = np.dot(dc, dc) / n
    for k in range(1, max_lag + 1):
        w = 1 - k / (max_lag + 1)            # Bartlett taper keeps the estimate >= 0
        lrv += 2 * w * np.dot(dc[k:], dc[:-k]) / n
    return d.mean() / np.sqrt(lrv / n), d.mean()


def align_forecast(r: pd.Series, h: pd.Series) -> pd.DataFrame:
    """Pair each forecast h[t] with the realised r[t+1]^2, indexed by the TARGET day t+1."""
    df = pd.DataFrame({"h": h.shift(1), "y": r**2})  # h made on day t-1 now sits on day t
    return df.dropna()
