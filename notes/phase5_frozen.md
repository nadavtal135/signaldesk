# Phase 5: frozen choice (committed BEFORE the final test is run)

**Frozen model: C3, the asymmetric EWMA.** It has 2 fitted parameters (λ, a),
plus a level-correction factor c that is derived from past data, not fitted.

    x[t] = r[t]^2 * (1 + a * [r[t] < 0])
    h[t] = c * ( λ * h[t-1]/c + (1-λ) * x[t] )      i.e. h = c * EWMA_λ(x)

The pipeline is frozen: fitted by minimizing QLIKE on an expanding window of past
data only, re-fitted every 63 trading days (first fit after 756 days), using
L-BFGS-B from the fixed start (λ = 0.94, a = 0.5) with bounds λ ∈ [0.80, 0.999]
and a ∈ [0, 3]. Re-fits keep running on that schedule; that is part of the
frozen pipeline, not tuning.

## Validation, 2022–2023 (501 days × 6 tickers, pooled across tickers)

| Model | Params | Pooled QLIKE | Gain vs C1 | DM t vs C1 | DM t vs GJR |
|---|---|---|---|---|---|
| C1 EWMA 0.94 | 0 | 2.1205 | — | — | −0.52 |
| C2 two-speed | 3 | 2.1485 | −0.028 | +2.20 | +1.15 |
| **C3 asymmetric** | **2** | **2.1116** | **+0.009** | **−1.66** | **−1.11** |
| C4 two-speed+asym | 4 | 2.1323 | −0.012 | +0.68 | +0.29 |
| C5 pull-to-level | 2 | 2.1392 | −0.019 | +1.61 | +0.56 |
| GJR-GARCH(1,1)-t | 5 | 2.1301 | −0.010 | +0.52 | — |
| Constant | 1 | 2.2635 | −0.143 | +3.33 | +2.87 |

How the rule was applied:
- C3 had the lowest validation QLIKE among C2–C5.
- No candidate has fewer parameters than C3. C5 also has 2, so it is not "simpler".
- C3 beats C1 on validation (+0.009), so C1 is not the fallback.

**Validation is not a strong signal.** C3's edge over C1 is small and not
significant (t = −1.66), and only one of the four extensions beat the plain
EWMA at all. The more flexible candidates (C2, C4, C5) did *worse* than C1 on
data they weren't fitted on, which is overfitting showing up in plain sight. As
in Phase 4, the plain EWMA also beat GJR-GARCH in 2022–2023.

## What each parameter does physically

- **λ (pole position):** sets the memory. A typical fitted λ of about 0.92–0.97
  gives a time constant of about 12–33 days.
- **a (down-day input gain):** a down day's r² enters the filter (1 + a) times
  as strongly as an up day of the same size. This is a polarity-dependent
  envelope detector, i.e. the "leverage effect". Median fitted a ranges from
  0.6 (SPY, XOM) to 1.2 (AAPL, JPM). For AMZN it sits at the upper bound of 3,
  a flag that the fit is pushing against its limit there.
- **c (level correction, derived):** rescales the output so its average equals
  the average r² over the fitting window. Without it, amplifying down days
  would bias every forecast upward.

## Other fitting observations (re-fits up to 2023 only)

These are flags of overfitting or misspecification:
- **C2:** puts about 70–85% weight on a fast pole (λf ≈ 0.6–0.8, a few days)
  and the rest on a slow one (λs ≈ 0.99).
- **C4:** a sits at the upper bound of 3 for 5 of 6 tickers.
- **C5:** λ sits at the lower bound of 0.80.
