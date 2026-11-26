# Phase 4 summary: is the simple EWMA as good as the professional models?

**Question:** does the one-line EWMA (λ = 0.94) forecast tomorrow's variance as
well as GARCH(1,1), GJR-GARCH(1,1) and a HAR-style regression? Variance is the
noise power of daily returns. All models use daily returns only, on the same six
tickers. Model descriptions are in [`phase4_models.md`](phase4_models.md).

## How it was tested

- **Fitting:** GARCH-t, GJR-t and HAR are fitted on an expanding window of past
  data only. The first fit uses 3 years of data; after that they re-fit every
  63 trading days. Between re-fits the forecast updates recursively with frozen
  parameters.
- **EWMA:** λ is frozen from Phase 2. The constant baseline is the 2015–2021 average.
- **Scoring:** every model is scored on exactly the same 2,236 days per ticker
  (2018 to today), with QLIKE (primary) and MSE, as in Phase 2. The test period
  is 2022 to today.
- **Significance:** Diebold-Mariano tests compare each model with the EWMA, per
  ticker and once on the six-ticker average. The tickers move together, so six
  per-ticker wins are not six independent confirmations. The pooled test is the
  honest single number.
- **Tests:** look-ahead for every model (including across a re-fit),
  hand-checked GARCH and GJR recursions, and strictly positive forecasts.
  All 39 tests pass.

## Results: test period, 2022 to today

QLIKE gain versus EWMA, where positive means the model beats the EWMA. The
**pooled DM t** below −2 means significantly better than the EWMA at about 95%.

| Model | Fitted params | Mean gain vs EWMA | Tickers better than EWMA | Pooled DM t | Runtime per ticker |
|---|---|---|---|---|---|
| EWMA λ=0.94 | 0 | — | — | — | 0.001 s |
| GARCH(1,1)-t | 4 | +0.018 | 5 of 6 | −1.9 | 0.7 s |
| **GJR-GARCH(1,1)-t** | 5 | **+0.031** | 4 of 6 | **−2.2** | 0.9 s |
| HAR-style | 4 | +0.003 | 3 of 6 | −0.2 | 0.006 s |
| Constant baseline | 1 | −0.087 | 0 of 6 | +2.5 (worse) | 0.001 s |

Per ticker, the significant wins over the EWMA are:
- GJR on SPY, AAPL and AMZN
- GARCH on AMZN
- HAR on AMZN

On **MSE**, no model is significantly different from the EWMA for any ticker
or in the pooled test. GJR lowers MSE by about 7% on SPY and AAPL, and is
slightly worse elsewhere. Full tables are in `figures/benchmark_test.csv` and
`figures/benchmark_complexity.csv`.

## Year by year (mean QLIKE gain vs EWMA over 6 tickers)

| | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026* |
|---|---|---|---|---|---|---|---|---|---|
| GARCH | +0.07 | +0.04 | +0.12 | −0.01 | −0.02 | −0.02 | +0.03 | +0.06 | +0.04 |
| GJR | +0.12 | +0.08 | +0.06 | +0.01 | −0.01 | −0.01 | +0.04 | +0.10 | +0.04 |
| HAR | +0.04 | +0.02 | +0.03 | +0.01 | −0.07 | −0.02 | +0.03 | +0.07 | +0.01 |

\*2026 runs January to November.

- **The GARCH-type models win most years, not every year.** They beat the EWMA in
  6 of 9 years. They lost, narrowly, in 2021–2023, so the test-period average is
  carried by 2024–2026.
- **Biggest wins come after sharp shocks.** The largest gains are in 2018 (the
  February volatility spike), 2020 (COVID) and 2025 (April). After a spike,
  GARCH-type forecasts jump faster and then relax back toward their long-run
  level. The EWMA has no long-run level, so it stays high for weeks
  (`figures/benchmark_spy_2025.png`).
- **In 2022, a long period of steadily elevated volatility,** that mean
  reversion slightly hurt: the models kept expecting calm that didn't come.

## Complexity vs. benefit, in plain words

- **Scale of the gain:** relative to the flat baseline, the EWMA captures a
  QLIKE gain of 0.087 on the test period. GJR-GARCH reaches 0.118, so the best
  professional model adds about **a third more** improvement on top of what the
  one-line filter already gets.
- **The price of that third:** five fitted parameters, an optimizer, quarterly
  re-fits and about 900× the runtime (still under a second per ticker).
- **Why GJR's extra parameter earns its keep:** down days raise future volatility
  more than up days. The fitted γ is large for SPY (0.35) and small for XOM
  (0.06), so the leverage effect is strongest in the index.
- **HAR doesn't pay off on daily data.** It tied with the EWMA on average and was
  the worst model in 2022. With only noisy daily r² as input, its three boxcar
  weights don't learn much that a single exponential filter misses.
- **Fits were stable:** all fits converged, GARCH persistence was 0.95–0.996, and
  the fitted tail parameter ν of 4–7 confirms heavy tails. The HAR positivity
  floor was never needed.

## Verdict

**GJR-GARCH(1,1) with Student-t errors is the best forecaster here, and the only
one that beats the EWMA with (borderline) statistical significance on the pooled
test (DM t = −2.2).** The margin is modest:
- about a third more improvement over the baseline than the EWMA already gets
- clearest for SPY, AAPL and AMZN
- concentrated in years with sharp shocks
- not visible at all on MSE

Plain GARCH is close behind (t = −1.9, not quite significant). HAR-style adds nothing.

So the **EWMA is a strong, cheap baseline, about 75% as good as the best model
on this yardstick, with zero fitted parameters.** It is not the best model. If
you can afford an optimizer, GJR-GARCH is a real but small upgrade.

## Caveats

- **The truth is a noisy proxy.** One day's r² is a very noisy measure of true
  variance, which makes every comparison noisy. Intraday "realized variance"
  would sharpen the test and is the natural input for HAR; it wasn't used, by design.
- **The test period is short.** 2022 to today is under 5 years, containing one
  big spike (April 2025). The year-by-year table shows the ranking flips in
  some regimes.
- **The tickers are correlated.** All six crash together, and the pooled DM
  test is the fair summary. Its t = −2.2 is borderline. Running several model
  comparisons also raises the chance that one looks significant by luck.
- **The baseline is in-sample for 2018–2021** (it uses the 2015–2021 average),
  and the EWMA's λ was also chosen on 2015–2021. Both are therefore slightly
  favoured in those years; the 2022-to-today numbers are fully out of sample
  for all models.
- **Simplifications:** zero mean for all models, one EWMA λ, quarterly re-fits,
  six large US names. Other choices could shift the margins.
- **Scope:** no trading logic, no machine learning, no new data (as instructed).
