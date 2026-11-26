"""Phase 5, step 5: null test for the frozen winner's pipeline.

Shuffle each ticker's returns within each period (2015-2021, 2022-2023,
2024-today), which keeps the values but destroys volatility clustering and any
down-day -> future-volatility link. Then re-run the FULL frozen pipeline (C3,
walk-forward re-fits) and the plain EWMA (C1) on the shuffled series, and
record the final-test pooled QLIKE gain of C3 over C1.

If C3's real gain over C1 came from extra flexibility alone (fitting noise),
the shuffled gains would be just as large. If it comes from real structure,
the real gain should sit above the shuffled distribution.

Run:  PYTHONPATH=src .venv/bin/python scripts/20_phase5_null_test.py
Outputs: figures/phase5_null.csv, figures/phase5_null.png
"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from signaldesk.data import TICKERS, load_prices
from signaldesk.ewma_family import CANDIDATES, c1_forecast, c3_path, walk_forward
from signaldesk.returns import log_returns

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
WIN = json.loads((FIG_DIR / "phase5_frozen_choice.json").read_text())["frozen_choice"]
BOUNDS = [("2015-01-01", "2021-12-31"), ("2022-01-01", "2023-12-31"), ("2024-01-01", "2026-12-31")]
TEST_START, TEST_END = "2024-01-01", "2026-11-25"
N_SHUFFLES = 200

# POST-HOC DIAGNOSTIC (not pre-registered, does not affect the verdict):
# the same C3 pipeline with the down-day gain a fixed at 0, i.e. only lambda is
# fitted. Comparing C3 against it isolates what the ASYMMETRY adds, separately
# from the freedom to re-fit lambda. It is registered here only for this script.
LAM_ONLY = "C3 with a=0 (diagnostic)"
CANDIDATES[LAM_ONLY] = (["lam", "a"], [0.94, 0.0], [(0.80, 0.999), (0.0, 0.0)], True, c3_path)


def daily_gains(r: pd.Series) -> pd.DataFrame:
    """Per-day QLIKE gains on final-test target days:
    vs_C1     = QLIKE(C1) - QLIKE(winner)          (pre-registered statistic)
    vs_lamonly = QLIKE(lambda-only) - QLIKE(winner) (post-hoc: value of the asymmetry)"""
    hw, _ = walk_forward(WIN, r)
    hl, _ = walk_forward(LAM_ONLY, r)
    h1 = c1_forecast(r)
    df = pd.DataFrame({"w": hw.shift(1), "l": hl.shift(1), "c1": h1.shift(1), "y": r ** 2}).dropna()
    df = df.loc[TEST_START:TEST_END]
    q = {k: df.y / df[k] + np.log(df[k]) for k in ["w", "l", "c1"]}
    return pd.DataFrame({"vs_C1": q["c1"] - q["w"], "vs_lamonly": q["l"] - q["w"]})


def pooled(gains: list[pd.DataFrame]) -> tuple[float, float]:
    """Average over tickers per day, then over days."""
    return tuple(float(pd.concat([g[c] for g in gains], axis=1).mean(axis=1).mean())
                 for c in ["vs_C1", "vs_lamonly"])


def shuffled(r: pd.Series, rng) -> pd.Series:
    parts = [rng.permutation(r.loc[a:b].values) for a, b in BOUNDS]
    return pd.Series(np.concatenate(parts), index=r.index)


def one_shuffle(seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    return pooled([daily_gains(shuffled(RETURNS[t], rng)) for t in TICKERS])


RETURNS = {t: 100 * log_returns(load_prices(t)["AdjClose"]) for t in TICKERS}

if __name__ == "__main__":
    real, real_l = pooled([daily_gains(RETURNS[t]) for t in TICKERS])
    with ProcessPoolExecutor(max_workers=4) as ex:
        res = np.array(list(ex.map(one_shuffle, range(N_SHUFFLES))))
    null, null_l = res[:, 0], res[:, 1]
    p = (1 + np.sum(null >= real)) / (1 + N_SHUFFLES)
    p_l = (1 + np.sum(null_l >= real_l)) / (1 + N_SHUFFLES)
    pd.DataFrame({"shuffle": range(N_SHUFFLES), "pooled_test_gain_vs_C1": null,
                  "posthoc_gain_vs_lambda_only": null_l}).to_csv(
        FIG_DIR / "phase5_null.csv", index=False, float_format="%.6f")
    summary = {"winner": WIN,
               "preregistered": {"real_gain_vs_C1": real, "null_mean": float(null.mean()),
                                 "null_p5": float(np.percentile(null, 5)),
                                 "null_p95": float(np.percentile(null, 95)),
                                 "p_value_real_above_null": float(p)},
               "posthoc_asymmetry_only": {"real_gain_vs_lambda_only": real_l,
                                          "null_mean": float(null_l.mean()),
                                          "null_p95": float(np.percentile(null_l, 95)),
                                          "p_value_real_above_null": float(p_l)}}
    (FIG_DIR / "phase5_null_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK2 = "#666666"
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e6e6", "axes.edgecolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.labelcolor": INK2, "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, nv, rv, pv, xl, tl in [
        (axes[0], null, real, p, "Gain over plain EWMA λ=0.94 (> 0 = C3 better)",
         "Pre-registered: C3 vs plain EWMA"),
        (axes[1], null_l, real_l, p_l, "Gain over the same pipeline with a = 0 (> 0 = asymmetry helps)",
         "Post-hoc diagnostic: what the down-day gain adds")]:
        ax.hist(nv, bins=30, color="#b3b3b3", edgecolor="white", linewidth=0.5,
                label=f"{N_SHUFFLES} shuffled return series")
        ax.axvline(rv, color="#257de0", linewidth=2, label="real data")
        ax.axvline(0, color=INK2, linestyle="--", linewidth=0.8)
        ax.set_xlabel(f"Final-test pooled QLIKE: {xl}")
        ax.set_title(f"{tl}   (p = {pv:.3f})", loc="left", fontsize=10, color="#191919")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    axes[0].set_ylabel("Number of shuffles")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase5_null.png", dpi=150)
