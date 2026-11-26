# Phase 5 summary: can an improved EWMA beat GJR-GARCH? (pre-registered)

**Result: no. The pre-registered rule says "did not beat GJR-GARCH."**
On the final test (2024 to 2026-11-25), GJR-GARCH was *significantly better*
than the frozen winner (pooled DM t = +2.24 against the winner), and won all
3 years. The improved EWMA did get better than the plain EWMA, and its one new
ingredient (down days count more) is real structure, not noise.

## The protocol, and why it matters

Everything was fixed in [`phase5_plan.md`](phase5_plan.md) and committed
**before** any Phase 5 model was run:
- the data split
- 5 candidates and their fitting procedure
- the selection rule
- the pass/fail rule

The validation pick was committed ([`phase5_frozen.md`](phase5_frozen.md))
**before** the final test was run, and the final test was run once. Nothing was
tuned afterwards. Git timestamps record the order. This protocol is what makes
a negative result trustworthy, and a positive one believable if it had happened.

- **Data split:**
  - fit walk-forward from 2018
  - validate on 2022–2023
  - final test on 2024 to 2026-11-25
- **Honesty note:** 2022 onward had already been seen in Phases 2–4, so it is
  "less seen", not fresh.

## Validation (2022–2023): picking one candidate

| Candidate | Fitted params | Pooled QLIKE gain vs plain EWMA |
|---|---|---|
| C2 two-speed | 3 | −0.028 (worse) |
| **C3 asymmetric** | **2** | **+0.009** (t = −1.66, not significant) |
| C4 two-speed + asymmetric | 4 | −0.012 (worse) |
| C5 pull to long-run level | 2 | −0.019 (worse) |

- **Pick:** the rule chose **C3, the asymmetric EWMA**: a down day's r² enters
  the filter (1 + a) times stronger, with a level correction so the average
  stays unbiased.
- **Overfitting was visible already here:** three of four extensions did
  *worse* than the plain EWMA on data they weren't fitted on, and several
  fitted parameters sat on their bounds.

## Final test (2024 to 2026-11-25), run once

| Model | Fitted params | Pooled QLIKE (lower is better) | Gain vs plain EWMA | DM t vs GJR |
|---|---|---|---|---|
| **GJR-GARCH(1,1)-t** | 5 | **1.784** | +0.059 (t = −2.93) | — |
| C3 asymmetric (frozen) | 2 | 1.826 | +0.017 (t = −1.94) | +2.24 (GJR better) |
| C1 plain EWMA λ=0.94 | 0 | 1.843 | — | +2.93 (GJR better) |
| Constant baseline | 1 | 1.891 | −0.048 | +1.83 |

- **Year by year:** GJR had the lowest pooled QLIKE in 2024, 2025 and 2026. The
  frozen C3 beat the plain EWMA every year.
- **Per ticker:** C3 beat GJR only on MSFT, and not significantly.
- **MSE:** no model differed significantly from any other.

**Pre-registered rule, applied literally:**
1. Pooled DM t < −2 against GJR? It is **+2.24 → fails**.
2. Lower QLIKE than GJR in at least 2 of 3 years? **0 of 3 → fails**.

**Verdict: did not beat GJR-GARCH.**

## Null test: is the improvement real structure or extra flexibility?

I shuffled the returns 200 times within each period. Shuffling keeps the values
but destroys volatility clustering and the link from down days to later volatility.
Then I re-ran the full fitting pipeline on each shuffled copy.

- **Pre-registered statistic** (C3's test gain over the plain EWMA): real
  +0.017 versus shuffled mean +0.092, so **p = 1.0**.
  - This statistic turned out to be a poor choice of yardstick. On shuffled
    data the best forecast is nearly a constant, so a model that can re-fit λ
    toward 0.999 easily beats a fixed λ = 0.94 that keeps chasing noise.
  - So "gain over the plain EWMA" mostly measures the freedom to re-fit λ, and
    it does not show that C3's real gain over the plain EWMA needs real structure.
- **Post-hoc diagnostic** (added after seeing the above, clearly labelled, and
  not part of the verdict): C3 against the *same* pipeline with the down-day
  gain fixed at a = 0. This isolates the asymmetry.
  - Real: +0.012. Shuffled: mean +0.0003, never above +0.010 in 200 tries, so **p = 0.005**.
  - **The down-day effect is real structure.** It vanishes when the time order
    is destroyed.

## Forward test: the only fully fair test

- **What's frozen:** C3 and GJR-GARCH parameters from their last re-fit (data up
  to 2026-11-25) are saved in `forward/frozen_model.json`.
- **First forecasts:** made on 2026-11-25, saved in `forward/forecasts.csv`.
- **How to run it later:** `scripts/score_forward.py` downloads fresh prices into
  `data/forward/` (the research CSVs stay untouched). It appends forecasts for
  each new trading day using only data up to that day, scores each forecast once
  the next day's return exists, and appends to `forward/score_log.csv`.
- **Append-only:** it never edits saved rows, and it warns if the data vendor
  revises history.
- **Why this is stronger than any backtest:** every backtest here was scored on
  data that existed while I was designing the models, and 2022 onward had been
  looked at in several phases. Forward days did not exist when the model was
  frozen, so no choice could have been influenced by them. That rules out data
  snooping and hidden look-ahead by construction.
- **When it means anything:** it needs months of data before it says anything
  (the scorer refuses a significance test under 30 days).

## What the numbers do and do not prove

**They do show:**
- **GJR-GARCH-t is the better daily variance forecaster for these six stocks
  over 2024–2026.** It beat both EWMA versions significantly.
- **Down days raising future volatility more than up days is real and
  exploitable inside an EWMA.** C3's asymmetry passed its null test (post-hoc).
- **Flexible extensions overfit.** Two-speed, combined and pull-to-level
  candidates all did worse than the plain EWMA on validation.

**They do not show:**
- **That C3 is significantly better than the plain EWMA.** Its t of −1.94 is
  just short of the bar.
- **That any EWMA variant can match GJR.** None did.
- **Anything about other assets, intraday data or future years.** That is what
  the forward test is for.

## Why GJR still wins, in signal-processing terms

GJR-GARCH has both features the EWMA lacks:
- **a polarity-dependent input gain** (C3 copied this one)
- **a set-point**, a long-run level the variance relaxes back to after a spike

The 2025 shock is where GJR gained the most: its forecast jumped faster and
then relaxed toward normal, while the EWMA-type filters stayed high for weeks.
The set-point candidate (C5) should have captured that, but it overfit and lost
on validation. 2022–2023 had no sharp spike to reward a set-point, and GJR
itself lost to the plain EWMA in those years. **Validation and test were
different regimes,** which is a limit of any single split.

## The useful finding, in plain words

- **How close it got:** with 2 fitted parameters, the asymmetric EWMA closed
  about **30% of the gap** between the plain EWMA and GJR-GARCH (5 parameters,
  plus an optimizer).
- **Why the gap remains:** the rest of it comes from structure the EWMA family
  here couldn't learn robustly, mainly mean reversion after shocks.
- **What it confirms:** the result supports Phase 4. GJR-GARCH is a real,
  modest upgrade, and it wasn't beaten by searching harder.

## Caveats

- **Multiple testing:** 5 candidates were tried. Picking the best of several on
  validation inflates its apparent edge, which is why the separate final test
  and strict rule were used. Earlier phases also looked at 2022 onward, so even
  the final test is not truly fresh.
- **The null-test statistic** was pre-registered but turned out to be
  confounded, as explained above. The diagnostic that fixes this was added
  after seeing results, so treat its p = 0.005 as exploratory, not confirmatory.
- **The tickers are correlated** (they crash together). The pooled test is the
  fair summary, not the per-ticker count.
- **The "truth" is a noisy proxy:** one day's r² is a very noisy measure of
  true variance.
- **Scope:** six large US stocks, daily data only, one 3-year final test with
  one big spike.
