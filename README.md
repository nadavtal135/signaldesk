# SignalDesk

A quant portfolio project that treats stock prices as **noisy signals** and asks,
with signal-processing tools, whether there is anything predictable in them.

Written from an RF / signal-processing point of view: every idea is explained in
DSP terms first, then translated into finance.

## Ground rules
- **No look-ahead:** a decision at time *t* may only use data up to *t*
  (the causal-filter rule). This gets enforced with tests once we start predicting.
- **Honest results:** "no detectable signal" is a valid outcome and will be reported as such.

## Layout
| Path | What lives there |
|------|------------------|
| `data/raw/` | Downloaded daily price CSVs |
| `src/signaldesk/` | Reusable library code |
| `scripts/` | One runnable script per analysis step |
| `figures/` | Generated plots |
| `notes/` | Written findings |
| `tests/` | pytest tests |

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Running Phase 1
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
```

## Phase 1 (done) - findings in [`notes/phase1_summary.md`](notes/phase1_summary.md)
1. Project setup
2. Download daily prices (SPY, AAPL, MSFT, AMZN, JPM, XOM; 2015 → today)
3. Daily returns, price vs. returns plots
4. Power spectral density of returns vs. white noise
5. Plain-English summary

**Headline:** daily returns are close to white noise (no linear signal for direction once the
2020 crash is excluded), but the noise *power* has strong, slow memory (volatility clustering).

## Phase 2 (done) - findings in [`notes/phase2_summary.md`](notes/phase2_summary.md)
Volatility tracker: causal low-pass filters (boxcar, EWMA) on squared returns, tuned on
2015-2021 and tested on 2022-today against a constant baseline, plus a shuffle null test.

**Headline:** an EWMA (lambda=0.94, ~16-day time constant) beats the constant baseline out of
sample on QLIKE for all six tickers, and a shuffle test confirms the edge comes from real
volatility clustering - but the improvement is modest, and on MSE it is not significant.
