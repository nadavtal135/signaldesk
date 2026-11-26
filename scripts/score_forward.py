"""Forward test scorer: run this any time after new trading days exist.

Two frozen sets, each with its own append-only logs in forward/:
- the original six (Phase 5):   frozen_model.json,        forecasts.csv,        score_log.csv
- the Phase 6 assets:           frozen_model_phase6.json, forecasts_phase6.csv, score_log_phase6.csv

For each set (append-only, never edits saved rows):
1. Downloads fresh daily prices (yfinance) into data/forward/ -- separate from
   the frozen research CSVs, which stay untouched. --no-download reuses them.
   BTC-USD is put on the US trading calendar (SPY's days), as in Phase 6.
2. For each trading day AFTER the last logged forecast, computes the frozen
   models' forecasts (fixed parameters, data up to that day only) and APPENDS them.
3. Scores every logged forecast whose target day (the next trading day) now
   has a return and has not been scored yet, APPENDING to the score log.
4. Warns if the data vendor revised history so a logged forecast no longer
   reproduces (the logged value always stands).
5. Prints the running pooled QLIKE of each model since the freeze date.

Why this is the strongest evidence: forward days did not exist when the
models were frozen, so no design choice could have been influenced by them.

Run:  PYTHONPATH=src .venv/bin/python scripts/score_forward.py [--no-download]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from signaldesk.data import download_prices, load_prices, save_prices
from signaldesk.evaluation import diebold_mariano
from signaldesk.forward import (FORECASTS, FORECASTS6, FROZEN, FROZEN6, MODELS, SCORES,
                                SCORES6, model_paths, qlike)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "forward"


def append(path: Path, df: pd.DataFrame):
    """Append rows; write a header only if the file is new. Never rewrites."""
    if len(df):
        df.to_csv(path, mode="a", header=not path.exists(), index=False, float_format="%.8g")


def prices(t: str, data_dir: Path, download: bool) -> pd.Series:
    if download:
        save_prices(download_prices(t), t, data_dir)
    p = load_prices(t, data_dir)["AdjClose"]
    if t == "BTC-USD":                      # sample bitcoin on the US trading calendar
        if not (data_dir / "SPY.csv").exists():
            save_prices(download_prices("SPY"), "SPY", data_dir)
        p = p.reindex(load_prices("SPY", data_dir).index).dropna()
    return p


def run_set(name: str, frozen_path: Path, forecasts_path: Path, scores_path: Path,
            data_dir: Path, download: bool):
    if not frozen_path.exists():
        print(f"[{name}] no frozen model at {frozen_path}; skipping")
        return
    frozen = json.loads(frozen_path.read_text())
    freeze = pd.Timestamp(frozen["freeze_date"])
    logged = pd.read_csv(forecasts_path, parse_dates=["made_on"])
    scored = pd.read_csv(scores_path, parse_dates=["made_on"]) if scores_path.exists() else None
    data_dir.mkdir(parents=True, exist_ok=True)

    new_fc, new_sc = [], []
    for t in frozen["params"]:
        r = 100 * np.log(prices(t, data_dir, download)).diff().dropna()
        h = model_paths(r, frozen["params"][t])
        mine = logged[logged.ticker == t]

        # (4) Integrity check against already-logged forecasts.
        for _, row in mine.iterrows():
            d = row.made_on
            if d in h.index and not np.allclose(h.loc[d, MODELS].values.astype(float),
                                                row[MODELS].values.astype(float), rtol=1e-3):
                print(f"[{name}] WARNING {t} {d.date()}: data revision changes this forecast; "
                      f"the logged value is kept.")

        # (2) New forecasts: days after the last logged one.
        last = mine.made_on.max()
        mine_new = [{"ticker": t, "made_on": d.date(), **{m: h.loc[d, m] for m in MODELS}}
                    for d in h.index[h.index > last]]
        new_fc += mine_new

        # (3) Score logged forecasts whose next trading day now exists.
        done = set() if scored is None else set(scored[scored.ticker == t].made_on)
        for _, row in pd.concat([mine, pd.DataFrame(mine_new)]).iterrows():
            d = pd.Timestamp(row.made_on)
            if d in done or d < freeze:
                continue
            nxt = r.index[r.index > d]
            if len(nxt) == 0:
                continue                       # target day not available yet
            y = r.loc[nxt[0]] ** 2
            new_sc.append({"ticker": t, "made_on": d.date(), "target_date": nxt[0].date(),
                           "r2": y, **{f"qlike {m}": qlike(y, float(row[m])) for m in MODELS}})

    append(forecasts_path, pd.DataFrame(new_fc))
    append(scores_path, pd.DataFrame(new_sc))
    print(f"[{name}] appended {len(new_fc)} forecasts and {len(new_sc)} scores")

    # (5) Running summary since the freeze.
    if scores_path.exists():
        S = pd.read_csv(scores_path, parse_dates=["target_date"])
        P = S.groupby("target_date")[[f"qlike {m}" for m in MODELS]].mean()   # pooled per day
        print(f"[{name}] forward test since {freeze.date()}: {len(P)} target days")
        print(P.mean().round(4).to_string())
        if len(P) >= 30:
            for a in ["qlike C3 asymmetric", "qlike C1 EWMA 0.94"]:
                tstat = diebold_mariano(P[a].values, P["qlike GJR-GARCH(1,1)-t"].values)[0]
                print(f"  pooled DM t, {a[6:]} vs GJR: {tstat:+.2f}  (< -2: significantly better than GJR)")
        else:
            print("  (fewer than 30 days: too early for a significance test)")


def main(download: bool):
    run_set("original six", FROZEN, FORECASTS, SCORES, DATA_DIR, download)
    run_set("phase 6 assets", FROZEN6, FORECASTS6, SCORES6, DATA_DIR, download)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    main(download=not ap.parse_args().no_download)
