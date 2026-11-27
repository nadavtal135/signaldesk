# Phase 8 pre-registration: a cleaner ruler (intraday realized variance)

Written and committed **before any intraday data was downloaded or computed**.
Nothing below may change after data or results are seen, except data drops made
by the fixed rules in (a), and each drop is logged with its reason.

**The point of this phase:** the models do not change. The yardstick does. Until
now a variance forecast was scored against one day's squared return, a single,
very noisy sample of the true power. Realized variance averages many intraday
samples and is a much less noisy truth.

**Terms:**
- **Realized variance (RV):** the sum of squared 5-minute returns within a day.
- **Overnight gap:** the move from one session's close to the next session's open.
- **HAR-RV:** a regression of tomorrow's RV on recent daily, weekly and monthly averages of RV.

## (a) Data

- **Instruments:** 5-minute bars for SPY, AAPL, MSFT, AMZN, JPM and XOM, regular
  US session only (09:30–16:00 New York time).
- **Source:** Alpaca market-data API (`data.alpaca.markets`). Bars are adjusted for
  splits and dividends (`adjustment=all`), from the earliest date the free tier
  serves (expected January 2016) to 2026-11-25.
- **Feed rule (fixed before seeing data):**
  - Use **SIP** (the consolidated feed, all US volume) if the free tier serves
    historical SIP bars over the whole range.
  - Otherwise use **IEX**. IEX carries only a few percent of US volume, so bars
    can be missing or unrepresentative for less-traded names. A missing bar is
    carried forward, so it contributes a zero return, which biases RV
    **downward**. Whichever feed is used is stated in the results.
- **Session calendar:** taken from Alpaca's calendar endpoint. A full day has 78
  bars; a 13:00 half-day has 42.

**Fixed quality rules:**
- **Day drops.** A ticker-day is dropped if:
  - more than 10% of its expected bars are missing, or
  - it has a broken print: a 5-minute move above 10% in absolute terms that
    reverses by more than half within the next bar.
- **Ticker drop.** A whole ticker is dropped if more than 5% of its days are
  dropped.
- **Splits** are adjusted by the vendor. This is checked at AAPL (2020-08-31) and
  AMZN (2022-06-06): no jump may appear in the overnight return on those days.

**Storage:** raw bars go in `data/intraday/`, which is gitignored and never
committed (vendor licensing). Only the download script and
`data/CHECKSUMS_intraday.sha256` are committed.

## (b) Target: close-to-close realized variance

For day d, with all log returns in %:

    r_on[d]  = 100 * ln( first trade price of session d / last trade price of session d-1 )
    r_i[d]   = 5-minute log returns within session d
    RV_cc[d] = r_on[d]^2 + sum_i r_i[d]^2
    r_cc[d]  = r_on[d] + sum_i r_i[d]          (= close-to-close log return)

**Why this definition:**
- **All models forecast close-to-close variance**, so the target must include the
  overnight gap. Session-only RV would penalize the daily-return models for a
  move they are supposed to forecast.
- **The daily return comes from the same bars.** So the squared-return input
  r_cc² and the RV input share exactly the same days and prices, and differ
  only in "square of the sum" versus "sum of the squares".
- **The overnight term is still a single noisy square.** Its share of RV is
  reported, but it is not rescaled, since rescaling would add a fitted parameter.

A day enters the target only if both day d and day d−1 pass the quality rules.

## (c) Models (frozen from earlier phases; nothing may be added)

1. **EWMA(r²):** EWMA, λ = 0.94, on r_cc². This is `signaldesk.ewma_family.ewma_filter` on r_cc².
2. **EWMA(RV):** the same EWMA, λ = 0.94, on RV_cc.
3. **GJR-GARCH(1,1)-t** on r_cc: `signaldesk.vol_models.garch_forecast(asymmetric=True)`.
   It uses the Phase 4 walk-forward schedule: first fit after 756 trading days,
   re-fit every 63 days, past data only.
4. **HAR-RV:** the Phase 4 HAR-style estimator (`signaldesk.vol_models.har_forecast`
   logic) fed with RV_cc instead of r².
   - Features are the 1-day, 5-day and 22-day averages of RV.
   - Weights are fitted by non-negative least squares, so forecasts are positive.
   - It uses the same walk-forward schedule as GJR.

The schedule means scoring starts about 3 years after the first data day
(expected January 2019). The exact scored range is reported.

## (d) Scoring

- **Losses:** QLIKE (RV/h + ln h) is primary and MSE is secondary. Both are
  computed against RV_cc of the target day d+1, for the forecast h made at the
  close of day d.
- **Identical days:** all four models are scored on exactly the same target days
  per ticker.
- **Pooling:** each day's loss is averaged over the 6 tickers, then
  Diebold-Mariano (Newey-West, 10 lags) is run on the pooled daily series. The
  tickers are correlated (they fall together), so per-ticker "wins" are not
  independent, and the pooled test decides.
- **Also reported:** per ticker, per year, and a complexity table (fitted
  parameters, runtime).

## (e) Hypotheses and decision rules

Each **holds** if t < −2, **fails** if t > +2, and is **inconclusive** otherwise.
Each t is a pooled DM statistic under the RV_cc target.

- **H1: an RV input beats a squared-return input.** t = DM of EWMA(RV) vs
  EWMA(r²). Only the EWMA exists in both versions, so H1 is judged on that pair.
- **H2: HAR-RV beats the EWMA fed with RV.** t = DM of HAR-RV vs EWMA(RV).
- **H3: with a cleaner target, GJR's edge over the EWMA disappears or flips.**
  t = DM of EWMA(r²) vs GJR.
  - Holds (the gap flipped: EWMA significantly better) if t < −2.
  - Fails (GJR still significantly better) if t > +2.
  - For context only, the same t is also reported under the old r_cc² target on
    the same days, so a *shrinking* gap is visible. The RV-target t decides.

## Null check (fixed now)

- **Method:** shuffle the daily order of each ticker's RV_cc series 200 times,
  which destroys persistence but keeps the values. Re-run the full HAR-RV
  walk-forward pipeline on each shuffled copy.
- **Statistic:** HAR-RV's pooled QLIKE gain over the **expanding mean of past
  RV**, a "no persistence" forecast that is part of this test statistic, not a model.
  - Lesson from Phase 5: comparing against the fixed EWMA on shuffled data
    would reward flexibility rather than detect persistence.
- **Expected:** real persistence should put the real gain above the shuffled
  distribution.

## (f) Honesty notes

- **2022 onward has already been seen.** Earlier phases scored models on those
  years with the old yardstick. New data does not undo that, and the forward
  test remains the only fully fresh evidence.
- **The feed limits apply.** If IEX is used, RV is biased low by missing bars,
  more so for less-traded names.
