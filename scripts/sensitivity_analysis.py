"""
sensitivity_analysis.py — Appendix A
=====================================
LSTM architecture sensitivity: hidden units × lags × dropout.
Produces TableA1_lstm_sensitivity.csv.

Usage
-----
    python scripts/sensitivity_analysis.py --currency EUR --segment asset_managers
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from itertools import product

from src.utils.config  import load_config
from src.utils.seed    import set_seed
from src.data.loader   import load_all
from src.models.lstm   import train_lstm
from src.forecasting.rolling_window import walk_forward
from src.data.preprocessing         import build_supervised
from src.portfolio.performance      import portfolio_stats, compute_portfolio_weights, portfolio_returns

ROOT = Path(__file__).resolve().parents[1]
cfg  = load_config(ROOT/"config/default.yaml", ROOT/"config/lstm.yaml",
                   ROOT/"config/portfolio.yaml")


def run_sensitivity(currency: str = "EUR", segment: str = "asset_managers"):
    set_seed(cfg.get("seed", 42))
    fx, of = load_all(cfg)

    hidden_units_grid = [16, 32, 64, 128]
    lags_grid         = [4, 8, 12]
    dropout_grid      = [0.10, 0.20, 0.30]

    rows = []
    baseline = (32, 8, 0.20)   # paper baseline

    for hu, lg, dr in product(hidden_units_grid, lags_grid, dropout_grid):
        # Only run baseline lags/dropout combinations to limit compute,
        # unless this is the full grid run
        run_cfg = {**cfg}
        run_cfg["lstm"] = {**cfg.get("lstm",{})}
        run_cfg["lstm"]["architecture"] = {
            **cfg.get("lstm",{}).get("architecture",{}),
            "hidden_units": hu,
            "dropout": dr,
        }
        run_cfg["preprocessing"] = {**cfg.get("preprocessing",{}), "lags": lg}

        def _trainer(X_tr, y_tr, c, seed=42):
            return train_lstm(X_tr, y_tr, run_cfg, seed=seed)

        try:
            result = walk_forward(
                fx=fx, of=of, currency=currency, segment=segment,
                oos_start=cfg["split"]["oos_start"], lags=lg,
                build_X_y=build_supervised, train_model=_trainer,
                cfg=run_cfg, model_name="lstm",
            )
            # Build single-currency portfolio
            pred_s = pd.Series(result.predictions, index=result.dates)
            act_s  = pd.Series(result.actuals,     index=result.dates)
            m_pred = pred_s.resample("ME").last()
            m_act  = act_s.resample("ME").sum()
            w = m_pred.apply(lambda v: max(v, 0))
            if w.sum() > 0: w = w / w.sum() * cfg.get("target_volatility", 0.10)
            ret = (w * m_act).dropna()
            st  = portfolio_stats(ret, annualise=12)
            is_baseline = (hu, lg, dr) == baseline
            rows.append({
                "hidden_units": hu, "lags": lg, "dropout": dr,
                "sharpe": round(st.sharpe, 3),
                "is_baseline": is_baseline,
            })
            flag = " ← BASELINE" if is_baseline else ""
            print(f"  hu={hu:3d} lag={lg:2d} dr={dr:.2f}  SR={st.sharpe:.3f}{flag}")
        except Exception as e:
            print(f"  FAILED hu={hu} lag={lg} dr={dr}: {e}")

    out = ROOT / "outputs" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "TableA1_lstm_sensitivity.csv", index=False)
    print(f"\nSaved TableA1_lstm_sensitivity.csv")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--currency", default="EUR")
    parser.add_argument("--segment",  default="asset_managers")
    args = parser.parse_args()
    run_sensitivity(args.currency, args.segment)
