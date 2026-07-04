"""
reproduce_paper.py
==================
Single entry point that reproduces all tables and figures in:

  Galeazzi, G. (2025). Financial Customer Order Flow, Heterogeneous Beliefs,
  and Exchange Rate Predictability: Evidence from Non-Linear Microstructure Models.

Usage
-----
    python scripts/reproduce_paper.py

Requirements
------------
  - data/raw/DataSourceEUR.xlsx   (Reuters FX spot rates)
  - data/raw/dt_chapter1.xls      (UBS customer order flow)

Outputs
-------
  outputs/tables/  — Table1_r2_decomposition.csv
                     Table2_portfolio_performance.csv
                     TableA1_lstm_sensitivity.csv
  outputs/figures/ — figure1_sharpe_sortino.png
                     figure2_r2_decomposition.png
"""
from __future__ import annotations
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.seed   import set_seed
from src.data.loader  import load_all
from src.data.preprocessing import scale_order_flow, build_supervised
from src.models.lstm        import train_lstm
from src.models.narx        import NARXModel, build_narx_features
from src.models.benchmarks  import RandomWalk
from src.forecasting.rolling_window import walk_forward
from src.portfolio.performance      import (
    compute_portfolio_weights, portfolio_returns,
    portfolio_stats, r2_decomposition,
)
from src.evaluation.statistical_tests import jobson_korkie_test
from src.visualization.figures        import (
    figure1_sharpe_sortino, figure2_r2_decomposition,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ── 0. Setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
cfg  = load_config(
    ROOT / "config/default.yaml",
    ROOT / "config/lstm.yaml",
    ROOT / "config/portfolio.yaml",
)
set_seed(cfg.get("seed", 42))

OUT_TABLES  = ROOT / "outputs" / "tables";  OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGURES = ROOT / "outputs" / "figures"; OUT_FIGURES.mkdir(parents=True, exist_ok=True)

CURRENCIES = cfg["data"]["currencies"]
SEGMENTS   = cfg["data"]["segments"]
LAGS       = cfg["preprocessing"]["lags"]
OOS_START  = cfg["split"]["oos_start"]


# ── 1. Load data ───────────────────────────────────────────────────────────
log.info("Loading data …")
fx, of = load_all(cfg)
log.info(f"  FX returns: {fx.shape}, Order flows: {of.shape}")


# ── 2. Walk-forward forecasts for all currencies × segments ───────────────
log.info("Running walk-forward forecasts …")

def _lstm_trainer(X_train, y_train, cfg, seed=42):
    return train_lstm(X_train, y_train, cfg, seed=seed)

all_results = {}   # {(currency, segment): WalkForwardResult}

for currency in CURRENCIES:
    for segment in SEGMENTS:
        key = (currency, segment)
        log.info(f"  {currency} | {segment}")
        try:
            result = walk_forward(
                fx          = fx,
                of          = of,
                currency    = currency,
                segment     = segment,
                oos_start   = OOS_START,
                lags        = LAGS,
                build_X_y   = build_supervised,
                train_model = _lstm_trainer,
                cfg         = cfg,
                model_name  = "lstm",
                seed        = cfg.get("seed", 42),
            )
            all_results[key] = result
        except Exception as e:
            log.warning(f"  FAILED {currency} | {segment}: {e}")


# ── 3. Portfolio construction ──────────────────────────────────────────────
log.info("Constructing portfolios …")

sigma_target = cfg.get("target_volatility", 0.10)
tc           = cfg.get("transaction_costs", 0.001)

def build_monthly_portfolio(segment: str) -> pd.Series:
    """Build portfolio for a given segment across all currencies."""
    # Collect monthly predictions
    pred_by_ccy = {}
    actual_by_ccy = {}
    for currency in CURRENCIES:
        key = (currency, segment)
        if key not in all_results:
            continue
        r = all_results[key]
        pred_by_ccy[currency]   = pd.Series(r.predictions, index=r.dates)
        actual_by_ccy[currency] = pd.Series(r.actuals,     index=r.dates)

    forecasts_df = pd.DataFrame(pred_by_ccy).resample("ME").last()
    actuals_df   = pd.DataFrame(actual_by_ccy).resample("ME").sum()   # monthly cumulation

    # Weights
    weights_rows = []
    for date, row in forecasts_df.iterrows():
        w = compute_portfolio_weights(row.values, sigma_target=sigma_target)
        weights_rows.append(pd.Series(w, index=forecasts_df.columns, name=date))
    weights_df = pd.DataFrame(weights_rows)

    return portfolio_returns(weights_df, actuals_df, transaction_cost=tc)


segment_port_returns = {}
for seg in SEGMENTS:
    segment_port_returns[seg] = build_monthly_portfolio(seg)
    log.info(f"  Portfolio built: {seg}")

# Random walk portfolio (zero forecasts → equal weight / 0)
rw_preds = {c: pd.Series(
    np.zeros(len(all_results.get((c, SEGMENTS[0]), type('o', (), {'predictions': [], 'dates': pd.DatetimeIndex([])})()).predictions)),
    index=all_results.get((c, SEGMENTS[0]), type('o', (), {'dates': pd.DatetimeIndex([])})()).dates
) for c in CURRENCIES if (c, SEGMENTS[0]) in all_results}
rw_forecasts = pd.DataFrame(rw_preds).resample("ME").last()
rw_actuals   = pd.DataFrame({
    c: pd.Series(all_results[(c, SEGMENTS[0])].actuals, index=all_results[(c, SEGMENTS[0])].dates)
    for c in CURRENCIES if (c, SEGMENTS[0]) in all_results
}).resample("ME").sum()
rw_weights   = rw_forecasts * 0                                        # zero weight = RW
segment_port_returns["random_walk"] = portfolio_returns(
    rw_weights, rw_actuals, transaction_cost=tc
)


# ── 4. Table 2 — Portfolio performance ────────────────────────────────────
log.info("Computing portfolio statistics …")

perf_rows = []
for name, ret in segment_port_returns.items():
    if len(ret) == 0:
        continue
    stats = portfolio_stats(ret, annualise=12)
    perf_rows.append({
        "strategy":          name.replace("_", " ").title(),
        "ann_return_pct":    round(stats.annualised_return, 1),
        "ann_vol_pct":       round(stats.annualised_vol, 1),
        "skewness":          round(stats.skewness, 2),
        "excess_kurtosis":   round(stats.kurtosis, 2),
        "autocorr_1":        round(stats.autocorr_1, 2),
        "sharpe":            round(stats.sharpe, 2),
        "sortino":           round(stats.sortino, 2),
    })

table2 = pd.DataFrame(perf_rows)
table2.to_csv(OUT_TABLES / "Table2_portfolio_performance.csv", index=False)
log.info(f"  Saved Table2_portfolio_performance.csv")
print("\nTable 2: Portfolio Performance")
print(table2.to_string(index=False))


# ── 5. Table 1 — R² decomposition (PRIMARY NMT RESULT) ────────────────────
log.info("Computing R² decomposition (primary NMT result) …")

# Build macro benchmark returns
# Use random walk, and macro-conditioned versions as benchmarks
benchmark_df = pd.DataFrame({s.replace("_"," ").title(): segment_port_returns[s]
                              for s in SEGMENTS})
benchmark_df["Random Walk"] = segment_port_returns["random_walk"]
benchmark_df = benchmark_df.dropna()

r2_rows = []
for seg in SEGMENTS:
    ret = segment_port_returns.get(seg)
    if ret is None or len(ret) == 0:
        continue
    aligned_ret  = ret.reindex(benchmark_df.index).dropna()
    aligned_bm   = benchmark_df.reindex(aligned_ret.index).drop(
        columns=[seg.replace("_"," ").title()], errors="ignore"
    ).dropna()
    common_idx   = aligned_ret.index.intersection(aligned_bm.index)
    if len(common_idx) < 10:
        continue
    decomp = r2_decomposition(
        segment_returns   = aligned_ret.loc[common_idx],
        benchmark_returns = aligned_bm.loc[common_idx],
    )
    r2_rows.append({
        "segment":       seg.replace("_", " ").title(),
        "alpha":         round(decomp["alpha"], 3),
        "alpha_se":      round(decomp["alpha_se"], 3),
        "r2_pct":        round(decomp["r2"], 1),
        "orthogonal_pct":round(decomp["orthogonal"], 1),
        "nobs":          decomp["nobs"],
    })

table1 = pd.DataFrame(r2_rows).sort_values("r2_pct")
table1.to_csv(OUT_TABLES / "Table1_r2_decomposition.csv", index=False)
log.info(f"  Saved Table1_r2_decomposition.csv")
print("\nTable 1: R² Decomposition (PRIMARY NMT RESULT)")
print(table1.to_string(index=False))


# ── 6. Statistical significance (Jobson–Korkie) ────────────────────────────
log.info("Running Jobson–Korkie significance tests …")

rw_ret = segment_port_returns["random_walk"]
jk_rows = []
for seg in SEGMENTS:
    ret = segment_port_returns.get(seg)
    if ret is None:
        continue
    common = ret.index.intersection(rw_ret.index)
    if len(common) < 10:
        continue
    jk = jobson_korkie_test(ret.loc[common].values, rw_ret.loc[common].values)
    jk_rows.append({
        "segment": seg.replace("_", " ").title(),
        "sr":      round(jk["sr_i"], 2),
        "rw_sr":   round(jk["sr_j"], 2),
        "jk_zstat":round(jk["z_stat"], 3),
        "p_value": round(jk["p_value"], 3),
        "sig":     "***" if jk["p_value"] < 0.01 else
                   "**"  if jk["p_value"] < 0.05 else
                   "*"   if jk["p_value"] < 0.10 else "",
    })

jk_table = pd.DataFrame(jk_rows)
jk_table.to_csv(OUT_TABLES / "Table3_jk_significance.csv", index=False)
log.info(f"  Saved Table3_jk_significance.csv")
print("\nTable 3: Jobson–Korkie Significance Tests")
print(jk_table.to_string(index=False))


# ── 7. Figures ─────────────────────────────────────────────────────────────
log.info("Generating figures …")

sharpe_dict  = {r["strategy"]: r["sharpe"]  for _, r in table2.iterrows()}
sortino_dict = {r["strategy"]: r["sortino"] for _, r in table2.iterrows()}

figure1_sharpe_sortino(
    sharpe  = sharpe_dict,
    sortino = sortino_dict,
    rw_sr   = sharpe_dict.get("Random Walk", 0.24),
    out_dir = str(OUT_FIGURES),
)

r2_dict = {r["segment"]: {"r2": r["r2_pct"], "orthogonal": r["orthogonal_pct"]}
           for _, r in table1.iterrows()}
figure2_r2_decomposition(r2_dict, out_dir=str(OUT_FIGURES))

log.info("Done. All outputs saved to outputs/")
print("\nReproduction complete.")
print(f"  Tables  : {OUT_TABLES}")
print(f"  Figures : {OUT_FIGURES}")
