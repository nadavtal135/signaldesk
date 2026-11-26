"""Forward-test scorer: simulate new trading days arriving after a freeze."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from signaldesk.data import RAW_DIR, TICKERS, load_prices, save_prices

ROOT = Path(__file__).resolve().parents[1]


def _load_scorer():
    spec = importlib.util.spec_from_file_location("score_forward", ROOT / "scripts" / "score_forward.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not (RAW_DIR / "SPY.csv").exists(), reason="needs price data")
def test_forward_scorer_appends_only_and_scores_once(tmp_path):
    sf = _load_scorer()
    frozen_real = json.loads((ROOT / "forward" / "frozen_model.json").read_text())
    # Pretend we froze on 2026-11-18 (same parameters), with data only up to then.
    freeze = pd.Timestamp("2026-11-18")
    frozen = {**frozen_real, "freeze_date": str(freeze.date())}
    (tmp_path / "frozen.json").write_text(json.dumps(frozen))
    data_dir = tmp_path / "data"
    from signaldesk.forward import MODELS, model_paths
    from signaldesk.returns import log_returns
    first = []
    for t in TICKERS:
        p = load_prices(t)
        save_prices(p.loc[:freeze], t, data_dir)
        r = 100 * log_returns(p.loc[:freeze]["AdjClose"])
        h = model_paths(r, frozen["params"][t])
        first.append({"ticker": t, "made_on": freeze.date(), **{m: h.loc[freeze, m] for m in MODELS}})
    pd.DataFrame(first).to_csv(tmp_path / "fc.csv", index=False, float_format="%.8g")

    def run():   # only temporary files: the real forward logs are never touched by tests
        sf.run_set("test", tmp_path / "frozen.json", tmp_path / "fc.csv",
                   tmp_path / "scores.csv", data_dir, download=False)

    # New data arrives: 2026-11-19 .. 2026-11-25 (5 trading days).
    for t in TICKERS:
        save_prices(load_prices(t), t, data_dir)
    before = (tmp_path / "fc.csv").read_text()
    run()
    fc = pd.read_csv(tmp_path / "fc.csv", parse_dates=["made_on"])
    sc = pd.read_csv(tmp_path / "scores.csv", parse_dates=["made_on", "target_date"])
    assert (tmp_path / "fc.csv").read_text().startswith(before)      # old rows untouched
    assert len(fc) == 6 * 6                                           # freeze day + 5 new days
    assert len(sc) == 6 * 5                                           # 5 forecasts now have a target
    assert (sc.target_date > sc.made_on).all() and (sc.made_on >= freeze).all()

    # Running again with no new data appends nothing.
    run()
    assert len(pd.read_csv(tmp_path / "fc.csv")) == 36
    assert len(pd.read_csv(tmp_path / "scores.csv")) == 30

    # A logged forecast is scored against the realised r^2 of the next day.
    p = load_prices("SPY")["AdjClose"]
    r = 100 * np.log(p / p.shift(1))
    row = sc[(sc.ticker == "SPY") & (sc.made_on == freeze)].iloc[0]
    assert np.isclose(row.r2, r.loc[row.target_date] ** 2)
