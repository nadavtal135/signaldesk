"""Risk targeting: size the position inversely to forecast volatility.

DSP view
--------
This is an automatic gain control (AGC) loop. The volatility tracker from
Phase 2 measures the signal level (noise power) of the stock's returns; we set
a gain = target_level / measured_level so the output (the strategy's returns)
has a roughly constant power. A gain ceiling (max_exposure) stops the gain
from running away when the input goes quiet. The gain is computed from data
up to day t and applied to day t+1, i.e. it is a causal, one-step-ahead AGC.

Finance view
------------
*Exposure* = the fraction of the account invested in the stock; the rest sits
in cash earning 0% here. Exposure 1.0 = all-in; 0.5 = half in the stock, half
in cash (so a +2% day for the stock is +1% for the account); 1.5 = 150%,
i.e. borrowing an extra 50% (leverage). Exposure is never negative (we never
bet on the price falling: "long only").

*Target volatility* = the annual volatility we would like the account to have.
If the stock's forecast volatility is 30%/yr and the target is 15%/yr, we hold
15/30 = 0.5 of the account in the stock.
"""
import numpy as np
import pandas as pd

from signaldesk.volatility import ewma_variance

TRADING_DAYS = 252
LAMBDA = 0.94   # frozen from Phase 2 (chosen on 2015-2021 only)


def forecast_vol_annual(log_r: pd.Series, lam: float = LAMBDA) -> pd.Series:
    """Annualised volatility forecast for day t+1, made at the close of day t.

    EWMA gives a DAILY variance h[t]. Daily noise samples add up in power over
    a year, so annual variance = 252 * h and annual volatility = sqrt(252 * h).
    """
    return np.sqrt(TRADING_DAYS * ewma_variance(log_r, lam))


def exposure_from_vol(vol_ann: pd.Series, target_vol: float, max_exposure: float = 1.0) -> pd.Series:
    """exposure[t] = min(max_exposure, target_vol / vol_ann[t]).

    Volatility is always positive, so the ratio is positive: long only by
    construction. NaN during the EWMA warm-up (no forecast yet -> no position).
    """
    return (target_vol / vol_ann).clip(upper=max_exposure)


def strategy_returns(simple_r: pd.Series, exposure: pd.Series) -> pd.Series:
    """Daily account return: exposure set at close of day t-1 times the stock's
    simple return on day t. Uninvested cash earns 0%.

    The .shift(1) is the whole no-look-ahead rule in one place: today's P&L
    uses yesterday's exposure decision.
    """
    return exposure.shift(1) * simple_r
