"""Phase 6, step 5: null check for H2 (pre-registered: BAC, GLD, BTC-USD; 100 shuffles each).

Shuffle each asset's returns within each period (2015-2021, 2022-2023,
2024-today): same values, but the time order -- including any link between a
down day and later volatility -- is destroyed. Rerun C3 and C3 with a = 0 and
record the 2018-today QLIKE gain of C3 over a = 0 (the "asymmetry gain").
If the real asymmetry gain is real structure, it should sit above the shuffles.

Run:  PYTHONPATH=src .venv/bin/python scripts/27_phase6_h2_null.py  [--plot-only]
Outputs: figures/phase6_h2_null.csv, figures/phase6_h2_null.json, figures/phase6_h2_null.png
"""
import os

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from signaldesk.ewma_family import CANDIDATES, c3_path, walk_forward
from signaldesk.universe import load_asset

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
ASSETS = ["BAC", "GLD", "BTC-USD"]
N_SHUFFLES = 100
LAM_ONLY = "C3 with a=0 (diagnostic)"
CANDIDATES[LAM_ONLY] = (["lam", "a"], [0.94, 0.0], [(0.80, 0.999), (0.0, 0.0)], True, c3_path)
BOUNDS = [("2015-01-01", "2021-12-31"), ("2022-01-01", "2023-12-31"), ("2024-01-01", "2026-12-31")]


def returns(t):
    return 100 * np.log(load_asset(t)).diff().dropna()


def asym_gain(r: pd.Series) -> float:
    """Mean over 2018-today target days of QLIKE(a=0) - QLIKE(C3)."""
    h3, _ = walk_forward("C3 asymmetric", r)
    h0, _ = walk_forward(LAM_ONLY, r)
    df = pd.DataFrame({"c3": h3.shift(1), "a0": h0.shift(1), "y": r ** 2}).dropna()
    df = df.loc["2018-01-01":"2026-11-25"]
    q = lambda h: df.y / h + np.log(h)
    return float((q(df.a0) - q(df.c3)).mean())


def one(job):
    t, seed = job
    r = returns(t)
    rng = np.random.default_rng(seed)
    parts = [rng.permutation(r.loc[a:b].values) for a, b in BOUNDS]
    return t, asym_gain(pd.Series(np.concatenate(parts), index=r.index))


def run():
    real = {t: asym_gain(returns(t)) for t in ASSETS}
    jobs = [(t, 1000 * i + s) for i, t in enumerate(ASSETS) for s in range(N_SHUFFLES)]
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(one, jobs))
    df = pd.DataFrame(res, columns=["ticker", "asym_gain"])
    df.to_csv(FIG_DIR / "phase6_h2_null.csv", index=False, float_format="%.6f")
    out = {}
    for t in ASSETS:
        nv = df[df.ticker == t].asym_gain.values
        out[t] = {"real_asym_gain": real[t], "null_mean": float(nv.mean()),
                  "null_p95": float(np.percentile(nv, 95)),
                  "p_value_real_above_null": float((1 + np.sum(nv >= real[t])) / (1 + len(nv)))}
    (FIG_DIR / "phase6_h2_null.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    df = pd.read_csv(FIG_DIR / "phase6_h2_null.csv")
    out = json.loads((FIG_DIR / "phase6_h2_null.json").read_text())
    INK2 = "#666666"
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for ax, t in zip(axes, ASSETS):
        ax.hist(df[df.ticker == t].asym_gain, bins=25, color="#b3b3b3", edgecolor="white",
                linewidth=0.5, label=f"{N_SHUFFLES} shuffles")
        ax.axvline(out[t]["real_asym_gain"], color="#257de0", linewidth=2, label="real data")
        ax.axvline(0, color=INK2, linestyle="--", linewidth=0.8)
        ax.set_title(f"{t}   p = {out[t]['p_value_real_above_null']:.3f}", loc="left", fontsize=10)
        ax.set_xlabel("QLIKE gain of C3 over C3 with a = 0\n(2018-today; > 0 = asymmetry helps)")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax.grid(True, color="#e6e6e6", linewidth=0.6)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Number of shuffles")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("H2 null check: is the down-day asymmetry gain real structure?",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG_DIR / "phase6_h2_null.png", dpi=150)


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run()
    plot()
