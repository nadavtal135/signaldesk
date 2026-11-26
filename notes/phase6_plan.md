# Phase 6 pre-registration: do the frozen models generalize to new assets?

Written and committed **before any new data was downloaded**. Models are frozen:
nothing is tuned or invented in this phase. The universe, decision rules and
hypotheses below may not change after data or results are seen. A data-quality
drop is allowed only by the fixed rules in (a), and every drop is logged with
its reason.

## (a) Universe: 26 assets (yfinance tickers), fixed now

| Class | Tickers |
|---|---|
| US stocks (15) | JNJ, PG, KO, WMT, CVX, BAC, GS, INTC, PFE, HD, CAT, UNH, NVDA, T, MRK |
| US equity ETFs (5) | QQQ (Nasdaq-100), IWM (small caps), XLE (energy), XLF (financials), XLU (utilities) |
| Bonds (1) | TLT (20+ year US Treasuries) |
| Gold (1) | GLD |
| International equity (2) | EFA (developed ex-US), EEM (emerging markets) |
| Oil (1) | USO (oil futures fund) |
| Bitcoin (1) | BTC-USD |

For H2, the "equity-type" group has 22 assets: the stocks, the equity ETFs, EFA and EEM.

**Fixed drop rules** (data quality only, never results). A ticker is dropped if:
- its first price is after 2015-01-02 (so data would start after 2018 once the
  3-year warm-up is counted), or
- it has a gap of more than 5 consecutive missing US trading days, or
- it has a non-positive price, or
- it has a one-day move above 50% in absolute terms that is not a verifiable
  real market move. That pattern is the signature of a split or data error.

Any drop is logged in `notes/phase6_data_checks.md` with its reason.

## (b) Data definitions

- **Source and dates:** yfinance adjusted close, daily, 2015-01-02 to 2026-11-25
  (the same end date as the original data), saved to `data/raw_phase6/`. The
  original six CSVs in `data/raw/` are not touched.
- **Returns:** r[t] = 100 × ln(P[t]/P[t−1]), as in Phases 2–5.
- **Bitcoin trades 7 days a week.** It is sampled on the **US trading calendar**:
  its close is taken on the days SPY trades. Weekend and holiday moves fold into
  the next trading day's return, just as weekend news does for stocks. So every
  asset has "one sample = one trading day", and λ = 0.94 and the 756/63-day
  re-fit schedule mean the same thing everywhere.
  - Caveat: Yahoo's BTC close is at 00:00 UTC, a few hours away from the US
    4 pm close, so its days are offset.
- **Other calendars:** international ETFs trade on US exchanges, so they already
  share the US calendar. Their underlying markets close earlier, which is a
  known timing offset and is noted, not corrected.

## (c) Frozen models (no code or parameter changes)

- **EWMA, λ = 0.94:** `signaldesk.ewma_family.c1_forecast`.
- **C3 asymmetric EWMA:** `walk_forward("C3 asymmetric", ...)`, with its fitting
  procedure unchanged. It is re-fitted walk-forward on each asset's own past data.
- **GJR-GARCH(1,1)-t:** `garch_forecast(..., asymmetric=True)`, with its
  walk-forward procedure unchanged.
- **Constant baseline:** each asset's own mean r² over 2015–2021. That makes it
  in-sample on 2018–2021 and fully out of sample on 2024 onward.
- **Diagnostic variant (H2 only, not a candidate):** C3 with a fixed at 0, so only
  λ is fitted. This is identical to the Phase 5 diagnostic. It isolates what the
  down-day asymmetry adds.

## (d) Scoring

- **Losses:** QLIKE is primary and MSE secondary. Each forecast h[t] is scored
  against r[t+1]². All models are scored on identical target days per asset,
  starting from the first day every model has a forecast (after the 756-day warm-up).
- **Periods:** 2018-01-01 to today (primary), and 2024-01-01 to today.
- **Pooling:** per day, average each model's loss over the assets, then run
  Diebold-Mariano (Newey-West, 10 lags) on the pooled daily series.
  - Assets are correlated (a market-wide shock hits most of them), so per-asset
    "wins" are not independent. The pooled test is the decision statistic.
  - QLIKE differences between models are scale-free (scaling all of an asset's
    returns changes every model's QLIKE by the same constant), so pooling
    assets with very different volatility is meaningful for model comparisons.
- **Breakdown:** per asset class (stocks, equity ETFs, bonds, gold,
  international, oil, bitcoin) and per asset.

## (e) Hypotheses and decision rules (written before any data)

**H1: GJR-GARCH beats plain EWMA on QLIKE across the pooled set.**
- Statistic: pooled DM t of GJR vs EWMA.
- **Holds** if t < −2. **Fails** if t > +2. **Inconclusive** otherwise.
- The 2018-today period decides. 2024-today is reported alongside and judged
  by the same rule.

**H2: the down-day asymmetry helps equities most, and gold, bonds and bitcoin least.**
- Statistic: per asset, the QLIKE gain of C3 over C3 with a = 0, on 2018-today.
- **Holds** if GLD, TLT and BTC-USD all fall below the median gain of the 22
  equity-type assets.
- **Fails** if 2 or more of GLD, TLT and BTC-USD are above that median.
- **Inconclusive** otherwise.

**H3: GJR's advantage over EWMA is larger for volatile assets (BTC, EEM) than calm ones (TLT, GLD).**
- Statistic: per asset, GJR's QLIKE gain over EWMA on 2018-today.
- **Holds** if both BTC-USD and EEM have a larger gain than both TLT and GLD.
- **Fails** if both TLT and GLD have a larger gain than both BTC-USD and EEM.
- **Inconclusive** otherwise.
- Also reported (descriptive, not decisive): the Spearman rank correlation, across
  all assets, between GJR's gain and the asset's annualized volatility over 2015–2021.

**Null check for H2:** for BAC (the equity, chosen now), GLD and BTC-USD, shuffle
returns within each period 100 times. Rerun C3 and C3 with a = 0, and compare
the real asymmetry gain (2018-today) with the shuffled distribution.

## (f) Known biases

- **Survivorship bias:** the stock list is today's large caps, i.e. companies that
  survived and grew. Failed or shrunk companies are missing, which likely
  understates how badly volatility can behave.
- **Data reuse:** these assets are new to the project, but the models and their
  design choices came from looking at the original six over the same years.
  Calendar effects shared across assets, like the 2020 and 2025 shocks, are not
  new information.
