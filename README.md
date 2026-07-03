# Microstructure Exchange Rate Forecasting

A fully reproducible implementation of the forecasting framework
presented in

> "Non-linear Forecasting using Machine Learning Models"

This repository reproduces the forecasting methodology developed in
the thesis using modern Python.

The repository contains

- complete preprocessing pipeline
- feedforward neural network
- LSTM forecasting
- NARX forecasting
- Random Walk benchmark
- Purchasing Power Parity benchmark
- Uncovered Interest Parity benchmark
- rolling out-of-sample forecasting
- portfolio construction
- Sharpe ratio
- Sortino ratio
- automatic reproduction of all figures and tables

---

microstructure_forecasting/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── .gitignore
├── config/
│   ├── default.yaml
│   ├── lstm.yaml
│   ├── narx.yaml
│   └── portfolio.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── interim/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_reproduce_chapter.ipynb
│   └── 04_portfolio_analysis.ipynb
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── preprocessing.py
│   │   ├── scaling.py
│   │   ├── supervised.py
│   │   └── validation.py
│   │
│   ├── models/
│   │   ├── feedforward.py
│   │   ├── lstm.py
│   │   ├── narx.py
│   │   ├── random_walk.py
│   │   ├── ppp.py
│   │   └── uip.py
│   │
│   ├── forecasting/
│   │   ├── trainer.py
│   │   ├── rolling_window.py
│   │   └── prediction.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── dm_test.py
│   │   ├── statistical_tests.py
│   │   └── diagnostics.py
│   │
│   ├── portfolio/
│   │   ├── optimizer.py
│   │   ├── allocation.py
│   │   ├── performance.py
│   │   └── risk.py
│   │
│   ├── visualization/
│   │   ├── figures.py
│   │   ├── tables.py
│   │   └── styles.py
│   │
│   └── utils/
│       ├── config.py
│       ├── logging.py
│       ├── seed.py
│       └── io.py
│
├── scripts/
│   ├── preprocess.py
│   ├── train_feedforward.py
│   ├── train_lstm.py
│   ├── train_narx.py
│   ├── reproduce_tables.py
│   ├── reproduce_figures.py
│   └── reproduce_paper.py
│
├── tests/
│
└── docs/


## Installation

```bash
git clone https://github.com/USERNAME/microstructure_forecasting.git

cd microstructure_forecasting

pip install -e .
```

or

```bash
pip install -r requirements.txt
```

---

## Data

Place the raw Excel files inside

```
data/raw/
```

```
DataSourceEUR.xlsx
dt_chapter1.xls
```

---

## Reproduce the paper

```
python scripts/reproduce_paper.py
```

---

## Repository

```
src/
    data/
    models/
    forecasting/
    evaluation/
    portfolio/
    visualization/
```

---

## Citation

If you use this repository please cite

Galeazzi (2023)

```
BibTeX
...
```

---

## License

MIT
