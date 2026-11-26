# Phase 6 data checks (step 2)

The universe was downloaded with yfinance (adjusted close, 2015-01-01 to
2026-11-25) into `data/raw_phase6/` and checked against the pre-registered drop
rules in `phase6_plan.md`. Full table: `figures/phase6_data_checks.csv`
(script: `scripts/24_phase6_data_checks.py`).

**Result: all 26 tickers pass. Nothing is dropped.**

| Rule | Result |
|---|---|
| First price on or before 2015-01-02 | All 26 (BTC-USD from 2015-01-01) |
| No run of more than 5 missing US trading days | 0 missing US trading days for every ticker (2,993 days each) |
| No non-positive prices, no missing values | 0 for every ticker |
| No one-day move above 50% (split or data error signature) | Largest is BTC-USD −46.5% on 2020-03-12, the real "Black Thursday" crash. All others are below 34%. |

Notable large moves were checked against known events. All are real:
- the March 2020 COVID crash (most tickers)
- USO −28.9% on 2020-04-21, the oil-futures collapse
- INTC −30.2% on 2024-08-02, after an earnings miss
- UNH −22% / −20% in 2025–2026
- NVDA +26% on 2016-11-11

**Splits:** NVDA (4:1 in 2021, 10:1 in 2024) and USO (1:8 reverse split in 2020)
show no jump in the adjusted series, so they are correctly adjusted. The day-to-day
change in the AdjClose/Close ratio is at most 3% for every ticker, which is
consistent with dividend adjustments only.

**Calendars:**
- **Bitcoin** has 4,347 daily closes (1,242 on weekends). It is sampled on the
  US trading calendar (SPY's days), so 1,354 non-US-trading days fold into the
  next US trading day's return. Every asset therefore has exactly the same 2,993
  sample dates.
- **International ETFs (EFA, EEM)** and **gold (GLD)** trade on US exchanges,
  so they are already on the US calendar. Their underlying markets close at
  different hours, a timing offset that is noted, not corrected.

**USO caveat (not a drop):** in April–May 2020 USO changed how it holds oil
futures (it moved away from the front month after the negative-price episode),
so its return series mixes two slightly different products. The data itself is
clean, so it stays, per the rules.
