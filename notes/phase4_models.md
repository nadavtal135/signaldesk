# Phase 4: the volatility models, in plain words

Every model below turns past daily returns r into a forecast h[t] of tomorrow's
variance (noise power), made at the close of day t. **Variance** is the noise
power of returns. **Volatility** is its square root.

## EWMA (the reference). 0 fitted parameters (λ = 0.94 frozen in Phase 2)
`h[t] = λ·h[t-1] + (1-λ)·r[t]²`

- **What it is:** a one-pole IIR low-pass filter on r², with unity gain at DC.
  It has a single pole at z = λ and no pull toward any long-run level: after a
  shock, the estimate decays toward wherever recent data sits.
- **How it was set:** λ was chosen once, on 2015–2021, so nothing is fitted
  here and it cannot overfit.

## GARCH(1,1) with Student-t errors. 4 parameters (ω, α, β, ν)
`h[t] = ω + α·r[t]² + β·h[t-1]`

- **What it is:** the same one-pole filter with a constant input ω added. That
  is a feedback filter with a **set-point**: the variance relaxes toward a
  long-run level ω/(1−α−β) at a rate set by the pole α+β. The EWMA is the
  special case ω = 0, α+β = 1 (a pole on the unit circle, so no set-point).
- **Student-t errors (ν):** the shocks may be heavy-tailed. ν is fitted, and a
  smaller ν means fatter tails. It does not change the forecast recursion; it
  only changes how ω, α and β are estimated.
- **Finance meaning:** the standard textbook volatility model. After a crash,
  volatility is expected to drift back to its normal level.

## GJR-GARCH(1,1) with Student-t errors. 5 parameters (ω, α, γ, β, ν)
`h[t] = ω + (α + γ·1[r[t] < 0])·r[t]² + β·h[t-1]`

- **What it is:** GARCH whose input gain depends on the sign of the input: a
  down-day feeds an extra γ·r² into the filter.
- **Signal-processing view:** an asymmetric (polarity-dependent) envelope detector.
- **Finance meaning:** the **leverage effect**, the empirical fact that falling
  prices raise future volatility more than rising prices of the same size.

## HAR-style regression. 4 parameters (b0, b1, b5, b22)
`h[t] = b0 + b1·r²[t] + b5·mean(r²[t-4..t]) + b22·mean(r²[t-21..t])`

- **What it is:** a bank of three boxcar low-pass filters (1-, 5- and 22-day
  windows: a day, a week and a month), mixed by weights learned with ordinary
  least squares. It is a coarse, 3-tap approximation of a long-memory filter:
  volatility "remembers" on several time scales at once.
- **Positivity:** weights are fitted with *non-negative* least squares (all
  b ≥ 0). The forecast is then a non-negative mix of non-negative averages, so
  it can't go negative. A tiny floor guards the all-zero corner case.
- **Name:** "HAR-style" because the original HAR model uses intraday "realized
  variance". Here we only have daily r², which is a much noisier input.

## Constant baseline. 1 parameter
- **What it is:** h = the average r² over 2015–2021 (as in Phase 2). It is a
  filter with infinite memory: it assumes the noise power never changes.

## Why more parameters can mean worse forecasts

Each fitted parameter is estimated from noisy data, and that estimation error
adds its own noise to every forecast. This is the same bias–variance trade as
choosing filter order: a higher-order filter can match the true response better,
but with a short, noisy training record its coefficients are themselves noisy.
Our "truth" (r²) is extremely noisy, so a flexible model can fit noise in the
training years that doesn't repeat. Fitting only on past data and scoring only
on future days, as here, is what exposes that.

## Fitting schedule (GARCH, GJR, HAR)

- **First fit:** after 756 trading days (3 years).
- **Re-fits:** every 63 trading days (about a quarter), each on an expanding window
  that uses only data up to and including the re-fit day.
- **Between re-fits:** the variance is updated day by day with the frozen
  parameters (HAR: its moving averages update with the same weights).
