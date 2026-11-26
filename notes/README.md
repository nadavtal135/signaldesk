# Notes: reading order

Each phase has a plain-English summary. Signal-processing terms come first, then
the finance meaning. Read them in this order:

1. [`phase1_summary.md`](phase1_summary.md): **Are daily returns white noise?** Their
   direction is (flat spectrum), but their power is not: volatility clusters for weeks.
2. [`phase2_summary.md`](phase2_summary.md): **A volatility tracker.** A one-pole
   low-pass filter (EWMA) on squared returns beats a constant forecast out of sample,
   modestly, and a shuffle test shows the edge is real.
3. [`phase3_summary.md`](phase3_summary.md): **Risk targeting.** Scaling exposure by
   1/forecast volatility gave a calmer account but no risk-adjusted edge, and a
   constant smaller position did about as well.
4. [`phase4_models.md`](phase4_models.md): **The professional models explained.**
   GARCH, GJR-GARCH and HAR described as filters, with parameter counts and the
   overfitting trade-off.
5. [`phase4_summary.md`](phase4_summary.md): **EWMA vs GARCH-family benchmark.**
   GJR-GARCH-t forecasts best, by a modest and borderline-significant margin; HAR adds nothing.
6. [`phase5_plan.md`](phase5_plan.md): **Pre-registration** of the Phase 5 data split,
   five candidates and pass/fail rule, committed before any model was run.
7. [`phase5_frozen.md`](phase5_frozen.md): **The validation pick** (asymmetric EWMA),
   frozen and committed before the final test.
8. [`phase5_summary.md`](phase5_summary.md): **Result: did not beat GJR-GARCH.** The
   pick improved on the plain EWMA and its down-day effect is real, but GJR stayed better.
9. [`phase6_plan.md`](phase6_plan.md): **Pre-registration** of the 26-asset universe,
   frozen models and hypotheses H1–H3, committed before any new data was downloaded.
10. [`phase6_data_checks.md`](phase6_data_checks.md): **Data-quality checks** for the
    new assets. All 26 passed, so none were dropped.
11. [`phase6_summary.md`](phase6_summary.md): **Generalization.** GJR's edge holds on
    new assets; the asymmetric EWMA's doesn't; the leverage effect shows up in equities only.

## A note on the price data

The daily prices come from Yahoo Finance via the `yfinance` package and are **not
included in this repository**, because Yahoo's terms are aimed at personal use.
`scripts/01_download_prices.py` and `scripts/23_phase6_download.py` download them, with
the end date pinned to 2026-11-25. `scripts/00_verify_data.py` checks the result against
`data/CHECKSUMS.sha256`, the checksums of the exact files every number in these notes
was computed from. Yahoo revises adjusted prices over time, so a later download may
differ slightly, and the results with it.
