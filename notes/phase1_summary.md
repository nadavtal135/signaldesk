# Phase 1 summary: are daily stock returns white noise?

**Data:** daily adjusted closing prices for SPY, AAPL, MSFT, AMZN, JPM and XOM,
2015-01-02 to 2026-11-25. That is 2,993 samples per ticker, with no missing
days. The *adjusted* price is corrected for stock splits and dividends (cash
paid to shareholders), so bookkeeping events don't show up as fake jumps.

**Signal model:** the price behaves like an integrator (a random walk) driven by
noise. Taking log-price and a first difference (a [1, −1] high-pass filter) gives
the **daily log return**, which is roughly the percent gain or loss each day. The
question is whether that "driving noise" is white.

## What the plots show

### 1. Price vs. returns (`figures/price_vs_returns_*.png`)
- Prices drift upward over 12 years. Returns look like zero-mean noise: the
  average day is +0.03% to +0.09%, buried under a 1.1–2.1% day-to-day spread.
  The trend ("DC term") is real but about 20× smaller than the noise on any single day.
- The noise is **bursty**. Quiet years alternate with loud episodes: March 2020
  (the COVID crash), 2022, and April 2025. Finance calls this **volatility
  clustering**. **Volatility** is the standard deviation of returns, which here
  is about 18%/year for SPY and 27–33%/year for the single stocks.
- The noise is **not Gaussian**. Days beyond 4σ happen 14–24 times per ticker,
  where a Gaussian would give about 0.2. Big spikes are market-wide: several
  tickers have their worst day on the same date.

### 2. PSD of returns (`figures/psd_returns.png`)
- **White noise** would give a flat spectrum: equal power at every frequency
  from DC up to Nyquist (0.5 cycles/day, i.e. a 2-day period). On the plot that
  is a line wandering around 0 dB inside the grey band. The band comes from
  shuffling the real data 500 times, which keeps the values but destroys any
  time order.
- The returns are **close to flat** (within about ±2 dB) for all six tickers.
- The one visible feature is a ~4 dB bump near 0.43 cycles/day in SPY, AAPL,
  MSFT and JPM. That is a small tendency for a day to partly reverse the
  previous one (lag-1 autocorrelation as large as −0.12). **It comes entirely
  from Feb–Jun 2020.** With that window removed the bump disappears and the
  lag-1 autocorrelation drops to between −0.04 and +0.02, which is not
  statistically significant once we allow for changing noise power.

### 3. PSD and autocorrelation of squared returns (`figures/psd_squared_returns.png`, `figures/acf.png`)
- Squaring the returns is like an envelope detector: it tracks the
  *instantaneous noise power*.
- This spectrum is **clearly not white**. Averaged over the five lowest
  frequency bins (cycles lasting about 50 to 256 trading days), it sits
  2.7 to 9.1 dB above the white level: SPY 9.1, JPM 8.5, XOM 6.9, AAPL 6.8,
  MSFT 6.3, AMZN 2.7 (`figures/whiteness_tests.csv`, column
  `low_freq_excess_squared_dB`). Single bins on the plot reach a little
  higher. The autocorrelation stays at 0.1–0.4 for more than 20 days. AMZN
  is the weakest case.

## Bottom line

| Question | Answer |
|---|---|
| Can past returns linearly predict the **direction** of tomorrow's return? | **No detectable signal.** Returns are white to within our measurement error, apart from one crash episode. |
| Can past returns predict the **size** of tomorrow's move (volatility)? | **Yes, clearly.** Noise power is a slowly varying envelope with memory lasting weeks. |
| Is the noise Gaussian? | **No.** It has heavy tails (rare huge spikes) and variance that changes over time. |

**Plain English:** these prices look like a random walk whose step size slowly
gets louder and quieter. You can't see which way the next step will go, but you
*can* tell roughly how big it is likely to be. Any later strategy that claims
to predict direction must beat this "white noise" baseline, and it must do so
outside the 2020 crash, not just because of it.

## Caveats
- These are *linear* tests (spectrum and autocorrelation). Non-linear structure
  could still exist and is not ruled out.
- Six large, heavily traded US stocks and ETFs over 12 years. Small or less-traded
  assets may behave differently.
- No prediction has been made yet, so no look-ahead risk exists. The only
  causal operation so far (computing returns) has a test that confirms it uses
  no future data.
