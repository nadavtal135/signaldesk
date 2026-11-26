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
```

## Phase 1 (in progress)
1. Project setup
2. Download daily prices (SPY, AAPL, MSFT, AMZN, JPM, XOM; 2015 → today)
3. Daily returns, price vs. returns plots
4. Power spectral density of returns vs. white noise
5. Plain-English summary
