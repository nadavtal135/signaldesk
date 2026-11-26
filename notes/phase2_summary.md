# Phase 2 summary: a volatility tracker

**Question:** Phase 1 found that the *direction* of daily returns behaves like
white noise, but their *power* (how big the moves are) drifts slowly. Can a
simple causal filter track that power well enough to forecast tomorrow's noise
power better than a flat, constant guess?

**Finance terms used:**
- **Variance** is the noise power of daily returns.
- **Volatility** is its square root, the typical size of a daily move.

**Data:** the same frozen CSVs as Phase 1 (six tickers, 2015-01-02 to 2026-11-25).

## Setup

- **What a forecast is:** a forecast made at the close of day *t* uses returns
  up to and including day *t* only. It is scored against the realised r[t+1]²,
  which is one very noisy sample of the true power.
- **The estimators, all low-pass filters on r²:**
  - **Boxcar** (moving average) with N = 10, 20 and 60 days. This is a finite-length filter with equal weights.
  - **EWMA** with λ = 0.94 and 0.97. This is a one-pole IIR filter with
    time constant ≈ 1/(1−λ), i.e. about 16 and 33 days (half-lives of about 11 and 23 days).
  - **Baseline:** a constant equal to the 2015–2021 average of r².
- **Scores:**
  - **MSE** is the squared error. It is dominated by a few crash days.
  - **QLIKE** is r²/h + ln h. It scores the *ratio* of truth to forecast, like
    dB, and penalizes under-forecasting risk. It is the standard, fairer score
    for volatility.
- **Honesty rules:**
  - N and λ were chosen on **2015–2021 only** and then frozen. 2022 to today was used only once, for the final score.
  - All estimators are scored on the same days.

## What worked

- **The EWMA beats the flat baseline out of sample, on QLIKE, for all six tickers.**
  - The tuned filter is λ = 0.94. Its test-period QLIKE gain is +0.03 to +0.20.
  - The gain is individually significant for SPY and XOM. For the other four it
    points the right way but is within the noise (`figures/vol_scores.csv`).
- **The improvement comes from real volatility clustering, not luck.**
  - After shuffling the returns (same values, time order destroyed), the same
    filter scores *worse* than the baseline in 5 of 6 tickers.
  - The real score beat all 200 shuffles for every ticker, so p ≤ 0.005
    (`figures/vol_null_test.png`).
  - The clustering is a genuine, detectable signal.
- **Smooth beats short.** The exponentially fading EWMA beats every boxcar on
  test. The 10-day boxcar is worse than the baseline on 5 of 6 tickers: its
  bandwidth is too wide, so it chases noise.

## What didn't (or only barely)

- **The edge shrinks a lot out of sample.** QLIKE gains fall roughly 2–10× from
  training to test (`figures/vol_scores_summary.png`). The training period contains
  the 2020 crash, where a flat line is badly wrong. In 2022 onward the average
  noise level stayed close to the training average, so tracking helps less.
- **On MSE the filters barely beat the baseline.** The best test improvement is
  +4.5% (SPY), and nothing is statistically significant. MSE is decided by a few
  extreme days, so it is too noisy a score to separate these forecasts.
- **Tracking has a cost.** When there is nothing to track (shuffled data) the
  EWMA adds its own estimation noise and loses to the constant. On real data it
  earns that cost back, but for MSFT and JPM the net gain is small.

## Bottom line

A one-pole low-pass filter on squared returns (EWMA, λ ≈ 0.94, about a 16-day time
constant) is a **real but modest** improvement over assuming the noise power
never changes. It helps most for the broad market (SPY) and in turbulent periods,
and least for single stocks in calm periods. The improvement is statistically
real because volatility clustering exists. In practical size it is small,
especially on 2022 to today.

## Caveats

- **The truth is a noisy proxy.** One day's r² is a very noisy measurement of the
  true power, so all scores are noisy. Intraday data would give a much cleaner
  "truth" but is not used here.
- **The baseline is slightly in-sample on training days** (it uses the
  2015–2021 average). On test days it is fully honest.
- **Only 2 λ values and 3 window lengths were tried.** A finer grid would fit
  the training data better but risks overfitting.
- **One train/test split.** A rolling, walk-forward evaluation would show how
  stable the edge is over time.
- **Significance tests are approximate.** Tickers are strongly correlated (all
  crash together), so six "wins" are not six independent confirmations. The
  Diebold-Mariano test uses an approximate correction for time correlation.
- **No GARCH and no trading logic** (both out of scope for this phase).
