# Phase 5 pre-registration: improving the EWMA without overfitting

This plan was written and committed **before any Phase 5 model was run or scored**.
Everything below is fixed in advance. Nothing may be added or changed after
results are seen. The git history timestamps this file.

## Goal

Find out whether a small, physically motivated extension of the EWMA volatility
filter can credibly beat GJR-GARCH(1,1)-t at forecasting tomorrow's variance.
"Credibly" is defined by the pass/fail rule in section (c), not by searching
until something wins.

## (a) Data split

- **Data:** the frozen daily CSVs (SPY, AAPL, MSFT, AMZN, JPM, XOM; 2015-01-02 to
  2026-11-25). Returns are daily log returns in %. Each forecast h[t] is made at
  the close of day t and scored against r[t+1]². Periods below refer to the
  **target day** t+1.

| Period | Dates | Use |
|---|---|---|
| Warm-up | 2015–2017 | Filter seeding and first fit only, never scored |
| Fit / select | 2018–2021 | Parameters are fitted walk-forward on expanding windows of past data |
| Validation | 2022–2023 | Used **once** to choose one candidate |
| Final test | 2024 to 2026-11-25 | Touched **exactly once**, after the choice is frozen |

- **Honesty note: 2022 onward is not fresh.** Phases 2–4 already scored models on
  2022 to today, and I saw those results. So the validation period is not
  independent of earlier choices, and 2024 onward is "less seen", not unseen.
  Only the forward test (step 6) is truly fresh.

## (b) Candidates (complete list, no additions later)

- **Input:** all candidates are causal filters on r².
- **EWMA filter:** EWMA_λ(x)[t] = λ·EWMA[t−1] + (1−λ)·x[t], seeded with the mean of
  the first 20 inputs (as in Phase 2).

| # | Name | Forecast h[t] | Fitted parameters |
|---|---|---|---|
| C1 | Baseline EWMA | EWMA_0.94(r²) (frozen from Phase 2) | 0 |
| C2 | Two-speed | w·EWMA_λf(r²) + (1−w)·EWMA_λs(r²) | λf, λs, w (3) |
| C3 | Asymmetric | c·EWMA_λ(x), with x = r²·(1 + a·1[r<0]) | λ, a (2) |
| C4 | Two-speed + asymmetric | c·[w·EWMA_λf(x) + (1−w)·EWMA_λs(x)] | λf, λs, w, a (4) |
| C5 | Pull to long-run level | (1−k)·EWMA_λ(r²) + k·m[t], with m[t] = mean(r²[0..t]) | λ, k (2) |

- **Level correction c (C3, C4):** inflating down days raises the average input,
  which would bias the forecast upward. So c = mean(r²) / mean(x), computed over
  the same past fitting window at each re-fit. It is derived from data, not a
  free parameter.
- **Long-run level m[t] (C5):** the expanding mean of all past r². It is causal:
  it uses r² up to day t only.

**Fitting (C2–C5):**
- **Objective:** minimize mean QLIKE = mean(r²[s+1]/h[s] + ln h[s]) over all
  pairs (s, s+1) with s+1 ≤ T. T is the re-fit day, so only past data is used.
  Scoring starts after the 20-day seed.
- **Optimizer:** L-BFGS-B from a fixed start. Bounds:
  - λ, λs ∈ [0.80, 0.999]
  - λf ∈ [0.50, 0.97]
  - w, k ∈ [0, 1]
  - a ∈ [0, 3]
  - λf < λs is enforced by fitting λs as λf + gap, with gap ∈ [0.001, 0.499]
    and λs capped at 0.999.
- **Fixed starts:**
  - λ = 0.94
  - λf = 0.90, λs = 0.98, w = 0.5
  - a = 0.5
  - k = 0.1
- **Schedule:** first fit after 756 trading days, then re-fit every 63 days on an
  expanding window, identical to Phase 4. Between re-fits the filter runs with
  fixed parameters.

**Comparison models (unchanged):**
- GJR-GARCH(1,1)-t from Phase 4 (`signaldesk.vol_models.garch_forecast`, asymmetric)
- the constant baseline (2015–2021 average r², from Phase 2)

## Selection rule (validation, applied once)

1. **Scoring:** score C1–C5 and GJR-GARCH on identical target days in 2022–2023,
   all six tickers.
2. **Pooled loss:** for each day, average the loss over the six tickers. Pooled
   QLIKE is the mean of that.
3. **Pick:** among C2–C5, take the lowest pooled validation QLIKE. If a candidate
   with *fewer* parameters is within noise of it (pooled DM |t| < 2), pick the
   simpler one. Ties on parameter count go to lower QLIKE.
4. **C1 can win:** if the picked candidate is not better than C1 on validation,
   the frozen choice is C1 and I report that no extension helped.
5. **Freezing:** the frozen choice (candidate and fitting procedure) is committed
   before the final test is run.

## (c) Pass/fail rule (final test, 2024 to 2026-11-25)

The frozen winner **passes** only if **both** hold:

1. **Pooled significance:** it beats GJR-GARCH on pooled QLIKE over the whole
   final test, with Diebold-Mariano t < −2 (Newey-West, 10 lags, as in Phase 4).
2. **Consistency:** it has lower pooled QLIKE than GJR-GARCH in at least 2 of
   the 3 test years: 2024, 2025 and 2026 (2026 runs January to 25 November).

Otherwise the result is **"did not beat GJR-GARCH"**, which is a valid result
and will be reported as such. MSE is reported as a secondary score but plays no
part in the rule.

## (d) Multiple testing

- **Candidates:** 5 (four extensions plus the baseline). Picking the best of
  several on validation inflates its apparent edge. That is why the final test
  is separate and the rule is strict.
- **Correlated tickers:** the six tickers move together, so per-ticker
  "wins" are not independent. The pooled test is the decision statistic.

## Further fixed steps

- **Null test:** shuffle each ticker's returns 200 times within each period
  (warm-up+fit, validation, test), which destroys volatility clustering. Then
  re-run the frozen winner's full fitting pipeline. The statistic is the winner's
  final-test pooled QLIKE gain over C1, computed on the same shuffled data. A
  real-structure improvement should sit well above the shuffled distribution.
- **Forward test:** freeze the winner's parameters from its last re-fit before
  2026-11-25, together with the code. Save forecasts with dates to
  `forward/forecasts.csv`. `scripts/score_forward.py` later appends forecasts for
  new days only (using data up to each day), scores them once the next day's
  return exists, and appends to `forward/score_log.csv`. Saved rows are never
  modified.
