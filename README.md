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
