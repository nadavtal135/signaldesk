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

**How the work is done:**
- **No look-ahead.** Every forecast at day *t* uses data up to *t* only,
  enforced by tests that perturb future prices and check that past outputs
  don't move (39 tests).
- **Out-of-sample discipline.** Parameters are tuned on 2015–2021 only and
  scored once on 2022 to today, or fitted walk-forward with expanding windows.
- **Controls, not just backtests.** Shuffle null tests, a constant-exposure
  placebo, and Diebold-Mariano tests with the correlated-tickers caveat stated.
- **Negative results are reported as results.** Phase 3's main conclusion is
  that the strategy's "safety" came from holding less, not from skill.

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
