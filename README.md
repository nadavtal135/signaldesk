# SignalDesk

A quant research project that treats stock prices as **noisy signals** and asks,
with signal-processing tools, what (if anything) is predictable in them.
Daily data for SPY, AAPL, MSFT, AMZN, JPM and XOM, 2015 to Nov 2026.

## Findings so far (2-minute read)

| Phase | Question | Answer |
|---|---|---|
| **1. Spectrum** | Is there a predictable signal in daily returns? | **Direction: no.** Returns have a flat spectrum, like white noise; the only exception came from the 2020 crash. **Size: yes.** Squared returns sit 3–10 dB above white at low frequencies, i.e. volatility clusters. |
| **2. Volatility tracker** | Can a causal filter forecast tomorrow's variance? | **Yes, modestly.** An EWMA (one-pole IIR, λ = 0.94) beats a constant forecast out of sample on all 6 tickers (QLIKE). A shuffle null test confirms the edge is real (p ≤ 0.005). On MSE the gain is not significant. |
| **3. Risk targeting** | Does sizing positions by 1/forecast vol beat buy-and-hold? | **Not risk-adjusted.** Drawdowns were smaller on every ticker (e.g. AMZN −32% vs −52%), but Sharpe matched buy-and-hold on 2022 to today, and a **constant smaller position with the same volatility did just as well**. The timing only helped in the 2020 crash. |
| **4. Model benchmark** | Is the EWMA as good as GARCH / GJR-GARCH / HAR? | **Close, but not the best.** GJR-GARCH-t wins on 2022 to today (pooled Diebold-Mariano t = −2.2, borderline), adding about a third more improvement over the baseline than the EWMA. It wins 6 of 9 years, mostly after sharp shocks. HAR on daily data adds nothing. No model beats the EWMA on MSE. |
| **5. Improved EWMA (pre-registered)** | Can a 2–4 parameter EWMA extension beat GJR-GARCH, without overfitting? | **No, by a rule written in advance.** 5 candidates were pre-registered, one was picked on 2022–2023 (asymmetric EWMA: down days count more), and the final test on 2024–2026 was run once. GJR was significantly better (t = +2.24 against the pick) in all 3 years. The pick closed about 30% of the EWMA-to-GJR gap with 2 parameters, and a shuffle test shows its down-day effect is real. A frozen forward test is set up for unseen days. |
| **6. New assets (pre-registered, models frozen)** | Do the models generalize to 26 new assets (15 stocks, 5 ETFs, bonds, gold, oil, international, bitcoin)? | **GJR's edge generalizes; the asymmetric EWMA's doesn't.** GJR beats the EWMA pooled (t = −3.09 over 2018 to today, −2.44 over 2024 to today; 22 of 26 assets). The fitted down-day effect is positive for equities and zero for gold, bonds and bitcoin, without being told which is which. GJR's edge tracks the leverage effect, not volatility: it barely helps bitcoin and loses on long bonds. |

**How the work is done:**
- **No look-ahead.** Every forecast at day *t* uses data up to *t* only,
  enforced by tests that perturb future prices and check that past outputs
  don't move (59 tests).
- **Out-of-sample discipline.** Parameters are tuned on 2015–2021 only and
  scored once on 2022 to today, or fitted walk-forward with expanding windows.
- **Controls, not just backtests.** Shuffle null tests, a constant-exposure
  placebo, and Diebold-Mariano tests with the correlated-tickers caveat stated.
- **Negative results are reported as results.** Phase 3's main conclusion is
  that the strategy's "safety" came from holding less, not from skill.
- **Pre-registration and a forward test.** Phase 5 fixed its candidates and
  pass/fail rule in a commit before running anything. It also logs frozen
  forecasts for days that didn't exist yet (`forward/`, `scripts/score_forward.py`).

Each phase has a plain-English write-up in `notes/`. Every idea is explained in
signal-processing terms first (filters, spectra, gain control), then in finance terms.

## Ground rules
- **No look-ahead:** a decision at time *t* may only use data up to *t* (the
  causal-filter rule), enforced with tests.
- **Honest results:** "no detectable signal" is a valid outcome and is reported as such.

## Layout
| Path | What lives there |
|------|------------------|
| `data/raw/` | Downloaded daily price CSVs (frozen snapshot) |
| `src/signaldesk/` | Library code: data, returns, spectral tools, volatility filters and models, evaluation, strategy, backtest |
| `scripts/` | One runnable script per analysis step |
| `figures/` | Generated plots and result tables |
| `notes/` | Plain-English findings per phase |
| `tests/` | pytest tests (look-ahead, hand-checked examples, calibration) |

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Reproduce
```bash
export PYTHONPATH=src
python scripts/01_download_prices.py   # download CSVs (needs Yahoo Finance network access)
python scripts/02_check_data.py        # sanity-check the samples
python scripts/03_returns_plots.py     # log returns, price-vs-returns plots, stats table
python scripts/04_psd.py               # PSD + autocorrelation vs. white noise (~1 min)
python scripts/05_vol_estimators.py    # Phase 2: causal volatility filters, SPY plot
python scripts/06_vol_evaluation.py    # Phase 2: train/test scores (MSE, QLIKE) vs constant baseline
python scripts/07_vol_null_test.py     # Phase 2: shuffle null test for the tuned estimator
python scripts/08_vol_summary_fig.py   # Phase 2: summary figure of train/test scores
python scripts/09_risk_target_demo.py  # Phase 3: risk targeting (AGC) illustration on SPY
python scripts/10_costs_turnover.py    # Phase 3: turnover and cost drag
python scripts/11_risk_target_eval.py  # Phase 3: tune target vol on train, test vs buy-and-hold
python scripts/12_placebo.py           # Phase 3: placebo (constant exposure) controls
python scripts/13_risk_target_null.py  # Phase 3: shuffled-forecast null test
python scripts/14_risk_target_plots.py # Phase 3: equity and drawdown figures
python scripts/15_model_benchmark.py   # Phase 4: EWMA vs GARCH / GJR-GARCH / HAR walk-forward benchmark
python scripts/16_benchmark_plots.py   # Phase 4: benchmark summary figures
python scripts/17_phase5_forecasts.py  # Phase 5: walk-forward forecasts for all candidates (cached)
python scripts/18_phase5_validation.py # Phase 5: validation 2022-2023 + pre-registered selection
python scripts/19_phase5_final_test.py # Phase 5: final test 2024-today + pass/fail rule
python scripts/20_phase5_null_test.py  # Phase 5: shuffle null test (~4 min)
python scripts/22_phase5_plots.py      # Phase 5: final-test figure
python scripts/23_phase6_download.py   # Phase 6: download the 26-asset universe (needs Yahoo access)
python scripts/24_phase6_data_checks.py
python scripts/25_phase6_run_models.py # Phase 6: frozen models on all assets (cached)
python scripts/26_phase6_results.py    # Phase 6: tables, hypotheses H1-H3, figure
python scripts/27_phase6_h2_null.py    # Phase 6: H2 null check (~1 min)
python scripts/score_forward.py        # Forward test: append + score new days (original six + Phase 6 set)
```

## Phase details

### Phase 1: returns as a signal ([`notes/phase1_summary.md`](notes/phase1_summary.md))
Download, sanity checks, log returns, and the power spectral density and autocorrelation of
returns and squared returns, compared against a shuffle-based white-noise band.

### Phase 2: volatility tracker ([`notes/phase2_summary.md`](notes/phase2_summary.md))
Causal low-pass filters (boxcar, EWMA) on squared returns, tuned on 2015–2021, tested on
2022 to today against a constant baseline with MSE and QLIKE, plus a shuffle null test.

### Phase 3: risk targeting ([`notes/phase3_summary.md`](notes/phase3_summary.md))
Exposure = min(cap, target vol / EWMA forecast vol), long only, 0.05% costs, compared
against buy-and-hold, a constant-exposure placebo and a shuffled-forecast null.

### Phase 4: model benchmark ([`notes/phase4_summary.md`](notes/phase4_summary.md), models explained in [`notes/phase4_models.md`](notes/phase4_models.md))
EWMA vs GARCH(1,1)-t, GJR-GARCH(1,1)-t and HAR-style, all walk-forward (expanding window,
re-fit every 63 days), scored on identical days with QLIKE, MSE, Diebold-Mariano tests and a
year-by-year breakdown.

### Phase 5: pre-registered EWMA improvements ([`notes/phase5_summary.md`](notes/phase5_summary.md), plan in [`notes/phase5_plan.md`](notes/phase5_plan.md))
Five EWMA-family candidates (two-speed, asymmetric, combined, pull to a long-run level), with the
split, selection rule and pass/fail rule committed before any model was run. Validation on
2022-2023 picked the asymmetric EWMA, and the single final test on 2024-2026 found it did **not**
beat GJR-GARCH. Also includes a null test and a frozen, append-only forward test.

### Phase 6: generalization to new assets ([`notes/phase6_summary.md`](notes/phase6_summary.md), plan in [`notes/phase6_plan.md`](notes/phase6_plan.md))
The frozen EWMA, C3 asymmetric EWMA and GJR-GARCH-t, run walk-forward on 26 new assets, with the
universe, hypotheses and decision rules committed before any data was downloaded. H1 (GJR beats
EWMA) holds, H2 (asymmetry matters for equities, not gold, bonds or bitcoin) holds but is small,
and H3 (the GJR edge grows with volatility) is inconclusive. The forward test now covers these
assets too.
