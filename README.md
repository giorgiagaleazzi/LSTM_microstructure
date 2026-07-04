# Neural Microstructure Technique (NMT)

> **Galeazzi, G. (2025). Financial Customer Order Flow, Heterogeneous Beliefs,
> and Exchange Rate Predictability: Evidence from Non-Linear Microstructure Models.**
> Working Paper, University of Glasgow.

This repository provides a fully reproducible implementation of the
**Neural Microstructure Technique** — a framework that combines disaggregated
customer order flow with LSTM recurrent neural networks to measure how much
of each customer segment's predictive content in FX markets is **orthogonal
to publicly observable macroeconomic information**.

The primary result is the **R² decomposition** (Table 1 / Figure 2): the
fraction of each segment's predictive content explained by standard macro
benchmarks. A low R² for asset managers (16.1%) and a high R² for corporates
(39.4%) is consistent with informed-trading theory.

---

## Repository structure

```
├── config/
│   ├── default.yaml          # data paths, currencies, sample period
│   ├── lstm.yaml             # LSTM architecture and training
│   └── portfolio.yaml        # portfolio construction parameters
│
├── src/
│   ├── data/
│   │   ├── loader.py         # load FX returns and UBS order flow
│   │   └── preprocessing.py  # scaling, supervised format, lags
│   ├── models/
│   │   ├── lstm.py           # LSTM build + train (paper Section 3.4)
│   │   ├── narx.py           # NARX robustness model
│   │   └── benchmarks.py     # Random Walk, UIP, PPP
│   ├── forecasting/
│   │   └── rolling_window.py # expanding walk-forward engine
│   ├── evaluation/
│   │   ├── metrics.py        # RMSE, MAPE, RMSFE ratio
│   │   └── statistical_tests.py  # Jobson–Korkie, Ledoit–Wolf, Lo SR CI
│   ├── portfolio/
│   │   └── performance.py    # SR, SO, R² decomposition (NMT primary)
│   └── visualization/
│       └── figures.py        # Figure 1, Figure 2
│
├── scripts/
│   ├── reproduce_paper.py    # single entry point — all tables + figures
│   └── sensitivity_analysis.py  # Appendix A LSTM architecture grid
│
└── outputs/
    ├── tables/               # CSV tables
    └── figures/              # PNG figures at 300 dpi
```

---

## Data

Place the raw data files in `data/raw/`:

| File | Description |
|---|---|
| `DataSourceEUR.xlsx` | Weekly FX spot rates (Reuters/DataStream) |
| `dt_chapter1.xls` | UBS customer order flow by segment |

> **Note:** UBS customer order flow data are proprietary and cannot be
> distributed. The sample covers November 2001 – November 2007 (317 weekly
> observations) and is identical to Cerrato, Sarantis & Saunders (2011) and
> Cerrato, Kim & MacDonald (2015).

---

## Installation

```bash
git clone https://github.com/giorgiagaleazzi/LSTM_microstructure.git
cd LSTM_microstructure
pip install -r requirements.txt
```

---

## Reproduce the paper

```bash
python scripts/reproduce_paper.py
```

This runs the full walk-forward exercise for all 9 currencies × 4 segments,
constructs portfolios, computes the R² decomposition (primary result),
runs Jobson–Korkie significance tests, and saves all tables and figures.

**Expected outputs:**

| File | Description |
|---|---|
| `outputs/tables/Table1_r2_decomposition.csv` | PRIMARY RESULT |
| `outputs/tables/Table2_portfolio_performance.csv` | SR, SO by segment |
| `outputs/tables/Table3_jk_significance.csv` | Jobson–Korkie p-values |
| `outputs/figures/figure1_sharpe_sortino.png` | Figure 1 |
| `outputs/figures/figure2_r2_decomposition.png` | Figure 2 |

## Architecture sensitivity (Appendix A)

```bash
python scripts/sensitivity_analysis.py --currency EUR --segment asset_managers
```

Produces `outputs/tables/TableA1_lstm_sensitivity.csv`.

---

## Citation

```bibtex
@techreport{galeazzi2025nmt,
  author      = {Galeazzi, Giorgia},
  title       = {Financial Customer Order Flow, Heterogeneous Beliefs,
                 and Exchange Rate Predictability:
                 Evidence from Non-Linear Microstructure Models},
  institution = {University of Glasgow},
  year        = {2025},
  type        = {Working Paper}
}

@phdthesis{galeazzi2023,
  author = {Galeazzi, Giorgia},
  title  = {Essays on International Economics},
  school = {University of Glasgow},
  year   = {2023}
}
```

---

## License

MIT
