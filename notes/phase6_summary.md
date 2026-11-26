# Phase 6 summary: do the frozen models generalize to 26 new assets?

**Setup:**
- **Universe:** the models were frozen from Phases 4–5, with no tuning and no
  new models, and run on 26 assets the project had never touched:
  - 15 US stocks
  - 5 equity ETFs
  - long Treasury bonds (TLT), gold (GLD) and oil (USO)
  - developed and emerging international stocks (EFA, EEM)
  - bitcoin
- **Pre-registration:** the universe, hypotheses and decision rules were
  committed in [`phase6_plan.md`](phase6_plan.md) before any data was downloaded.
- **Data checks:** all 26 assets passed ([`phase6_data_checks.md`](phase6_data_checks.md)), so nothing was dropped.

## Headline

| Hypothesis (written in advance) | Verdict by the pre-registered rule |
|---|---|
| **H1:** GJR-GARCH beats plain EWMA, pooled | **HOLDS.** Pooled DM t = −3.09 (2018 to today) and −2.44 (2024 to today) |
| **H2:** the down-day asymmetry helps equities most, and gold, bonds and bitcoin least | **HOLDS**, but the effect is small and concentrated in a few equities |
| **H3:** GJR's edge is larger for volatile assets (BTC, EEM) than calm ones (TLT, GLD) | **INCONCLUSIVE.** EEM fits; BTC doesn't. Volatility doesn't predict GJR's edge (rank correlation −0.04). |

## What generalized

**GJR-GARCH's advantage over the EWMA carries over to new assets.**

- **Pooled over all 26:**
  - QLIKE gain of +0.026 over 2018 to today (t = −3.09)
  - +0.022 over 2024 to today (t = −2.44)
  - That's about the same size as on the original six in Phase 4 (+0.031).
- **Per asset:** it beats the EWMA on 22 of 26 in both periods. That includes
  all 5 equity ETFs and both international funds.
- **By class (2018 to today, pooled within class):**
  - stocks t = −2.74
  - equity ETFs t = −2.53
  - international t = −2.90
- **On MSE,** as before, nothing is significant (pooled t ≈ −0.2).

**The leverage effect is real where finance says it should be, and absent
where it shouldn't be.** The fitted down-day gain tells the story without
being asked to:
- **Equities:** the C3 asymmetry parameter is positive in every fit for most
  stocks and equity ETFs. The median a is about 0.9: down days count about
  twice as much.
- **Gold, bonds and bitcoin:** a = 0 (the lower bound) in 97–100% of fits.
  - GJR's down-day term γ is even slightly *negative* for these three, i.e. up
    days raise their volatility a little more. That's the opposite of stocks.
- **BAC null check:** the asymmetry gain (+0.019) sits above all 100 shuffled
  versions (p = 0.010). It's real structure.
- **GLD and BTC null checks:** the gain is exactly 0, because a = 0, and it
  sits inside the shuffled spread (p ≈ 0.35). There's no asymmetry to find.

## What did not generalize

- **The Phase 5 asymmetric EWMA (C3) does not beat the plain EWMA on new
  assets.** Pooled, it is slightly *worse* over 2018 to today (−0.008, t = +0.75)
  and slightly better over 2024 to today (+0.006, t = −1.23). Neither is
  significant.
  - GJR beats it pooled (t = +2.37 against C3 over 2018 to today).
  - C3's small Phase 5 edge on the original six did not travel.
- **The asymmetry's forecasting value is small even where it is real.** The
  median gain across the 22 equity-type assets is +0.002 QLIKE. It is
  noticeable only for financials and broad indices:
  - BAC +0.019, IWM +0.013, XLF +0.011, EFA +0.010
  - several of these are borderline significant
  - several defensive stocks (JNJ, PFE, WMT) are slightly negative
- **"More volatile means a bigger GJR edge" is not supported.**
  - Bitcoin, the most volatile asset (74% a year), gains only +0.010 from GJR.
  - TLT, the calmest, is the one asset where GJR is clearly *worse* (−0.043).
  - The largest GJR gains are XLF (+0.081), BAC (+0.072), EFA (+0.066) and
    QQQ (+0.048). Those are leverage-effect assets, not high-volatility ones.

## What surprised me

1. **It's about the leverage effect, not volatility.** GJR's advantage tracks
   *where down days drive volatility* (banks, broad equity indices), not *how
   volatile* an asset is. Bitcoin is wild but symmetric, so the fancy model
   barely helps.
2. **Bonds broke GJR.** TLT over 2018 to today was the only clear loss for GJR.
   A likely reason (not tested here): the 2022 bond crash was a long, steady
   rise in volatility, the kind of regime where GJR's pull back to a long-run
   level expects calm that doesn't come. That matches the weakness seen in
   2022 for stocks in Phase 4.
3. **The model "discovered" asset-class physics on its own.** The fitted
   asymmetry is near 0 for gold, bonds and bitcoin and clearly positive for
   equities, with no labels given. That's evidence the fitting procedure
   measures something real rather than noise.

## Forward test, extended

- **What's frozen:** the 26 assets have their own frozen parameter set
  (`forward/frozen_model_phase6.json`, taken from each asset's last walk-forward
  fit with no re-fitting) and their own append-only logs (`forecasts_phase6.csv`,
  `score_log_phase6.csv`).
- **Running it:** `scripts/score_forward.py` now handles both the original six
  and the Phase 6 set in one run. It reads its ticker lists from the frozen
  files, and samples bitcoin on the US calendar.
- **Rounding:** the frozen parameters are the 5-decimal values saved from the
  walk-forward run. They reproduce the walk-forward forecasts on the freeze day
  to within 0.012%.

## Caveats

- **Correlated assets:** 26 assets are not 26 independent tests. A market-wide
  shock (2020, 2025) hits most of them on the same days. The pooled DM test is
  the honest summary, and per-asset counts like "22 of 26" overstate the evidence.
- **Survivorship bias:** the stocks are today's large caps, the survivors.
  Companies that collapsed are missing, and their volatility would be harder to forecast.
- **The years aren't new.** The assets are new, but 2018–2026 contains the same
  shocks the models were designed around. A truly new test needs new years,
  which is what the forward test is for.
- **Single-asset classes:** gold, bonds, oil and bitcoin each rest on one asset,
  so their verdicts are anecdotes, not class-level findings.
- **Timing offsets:** bitcoin is sampled on the US calendar from midnight-UTC
  closes, and the international funds' underlying markets close earlier than
  New York. Both add timing noise.
- **Limited H2 test:** H2 holds by the literal rule, but the rule only compares
  three non-equity assets against a median that is itself tiny (+0.002). It
  supports the *direction* of the hypothesis, not a large effect.
