# Phase 3 summary: risk targeting (volatility-scaled exposure)

**Question:** Phase 2 showed that tomorrow's noise power is partly predictable.
If we hold *less* stock when the forecast volatility is high and more when it
is low, do we get a better risk-adjusted result than simply holding the stock?

This is a backtest only: historical simulation, no broker, no live trading.

## The rule

- **Exposure:** at the close of day *t*, set
  exposure[t] = min(cap, target_vol / forecast_vol[t]), using the frozen EWMA
  (λ = 0.94) from Phase 2, annualized. That exposure earns the stock's return on
  day *t+1*.
- **Meaning of exposure:** the fraction of the account in the stock. The rest is
  cash at 0%. So 0.5 means half in the stock, and above 1 would be borrowed money.
  The strategy is long only.
- **Signal-processing view:** an automatic gain control loop. Measure the input
  level, set gain = target / level, and cap the gain.
- **Costs:** 0.05% of the amount traded each day.
- **Honesty rules:** target vol was chosen on **2015–2021 only** from {10, 12, 15, 20}%
  and then frozen (20% for cap 1.0, 10% for cap 1.5). 2022 to today was used once,
  for reporting.

## Results on the test period (2022 to today, cap 1.0, net of costs)

| | Sharpe: strategy / buy-and-hold | Max drawdown: strategy / buy-and-hold | Annual return: strategy / buy-and-hold |
|---|---|---|---|
| SPY | 0.64 / 0.64 | −23% / −25% | 9.1% / 10.2% |
| AAPL | 0.71 / 0.66 | −25% / −33% | 13.1% / 15.9% |
| MSFT | 0.40 / 0.44 | −28% / −36% | 6.3% / 8.9% |
| AMZN | 0.40 / 0.40 | −32% / −52% | 6.6% / 8.2% |
| JPM | 0.82 / 0.80 | −29% / −38% | 15.7% / 18.2% |
| XOM | 0.98 / 0.99 | −17% / −21% | 19.4% / 25.7% |

The **Sharpe ratio** is return per unit of volatility, i.e. the signal-to-noise
ratio of the P&L. The **max drawdown** is the worst fall from a previous peak.
Full tables are in `figures/risk_target_metrics.csv`, and the curves are in
`figures/risk_target_test_equity_drawdown.png` and `figures/risk_target_full_equity.png`.

## What worked

- **It delivered a calmer account.** On every ticker the strategy had lower
  volatility, smaller max drawdowns (AMZN −32% vs −52%, AAPL −25% vs −33%) and
  smaller worst days than buy-and-hold.
- **Costs were not the problem.** Turnover was about 1.5–5× the account per year,
  costing roughly 0.1–0.25% per year at 0.05% per trade. The gain moves almost
  every day, but in small steps.
- **In 2015–2021 it looked genuinely good.** Sharpe beat buy-and-hold on 5 of 6
  tickers. At equal risk it beat a constant-exposure control by about 1.3 points
  of return per year and had 3 points smaller drawdowns on average. It turned
  the gain down before the worst of the March 2020 crash.

## What didn't

- **There is no risk-adjusted edge out of sample.** On 2022 to today, Sharpe
  was within ±0.06 of buy-and-hold for every ticker. The strategy earned less,
  in roughly fair proportion to the risk it took.
- **The placebo explains the out-of-sample benefit.** A constant exposure with
  the same volatility and no timing earned about the same return (+0.03 points
  per year on average, 3 of 6 better) and had about the same drawdown
  (+0.2 points, 3 of 6 better). On the test period, the smaller drawdowns came
  from **being invested less, not from clever timing**.
- **The shuffle test agrees.** Scrambling *when* each forecast is applied made
  no difference to Sharpe (p = 0.27–0.60). Real timing gave a slightly smaller
  drawdown than the typical shuffle in all 12 cases (6 tickers × 2 caps), but
  only 2 were individually significant. That is suggestive, not proven.
- **Tuning was noise.** Mean training Sharpe varied only 0.950–0.959 across the
  target grid. The target sets *how much risk you take*, not *how good the
  result is*.
- **The 1.5 cap didn't help.** With its tuned 10% target it actually held less
  (average exposure 0.3–0.7). It behaved like a smaller version of the cap 1.0
  strategy, with slightly worse test Sharpe on 3 tickers.

## The tradeoff, in plain words

Risk targeting is **a volume knob with a slow automatic hand on it**. You give up
some return (1–6 points a year here) and get a smoother ride with shallower
drops. That is a legitimate choice for someone who would panic-sell in a −50%
drawdown. But over 2022 to today, you could have bought the same smoother ride
just by holding a fixed smaller position. The volatility timing itself added
little. Its one clear success, the 2020 crash, was a sudden spike that a
16-day filter can react to. The 2022 bear market was a slow grind that it
can't step around.

## Caveats

- **Cash earns 0% here.** In 2023–2025 cash actually paid about 4–5%. Every
  low-exposure strategy (risk target and placebo alike) is understated, by about
  1–2.5 points per year on its uninvested cash. This does not change the
  strategy-versus-placebo comparison.
- **Trades are assumed to fill at the same closing price used to make the decision.**
  In reality you'd trade slightly after, at a slightly different price.
- **Costs ignore the small extra rebalancing needed as prices drift between days.**
- **The placebo uses the period's own average exposure**, so it is a diagnostic
  control, not a strategy anyone could have run in advance.
- **One train/test split, six correlated US large-cap names, and one big crash in
  training.** These results may not hold for other assets or other periods.
- **No leverage costs** (borrowing for exposure above 1), no taxes, no slippage on large orders.

## Bottom line

Honest result: **risk targeting with an EWMA forecast did not beat buy-and-hold
on a risk-adjusted basis, and did not beat a constant-exposure placebo, over
2022 to today.** It reliably made the account calmer, but mostly by holding less.
The timing helped once, in a sudden crash in the training period, and showed at
most a faint drawdown benefit out of sample.
