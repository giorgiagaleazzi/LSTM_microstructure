"""
Preprocessing pipeline — paper Section 3.4 and Section 4.

Key steps
---------
1. Standardise order flows by training-set standard deviation.
   All scaling parameters estimated on training set only (no lookahead).
2. Build supervised learning format with k lags.
3. Split into train / validation / test sets.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class ScalingParams:
    of_std: pd.Series       # std dev of each order-flow column (training set)
    # FX returns are not scaled — they enter as log returns directly


def scale_order_flow(
    of: pd.DataFrame,
    train_end_idx: int,
) -> tuple[pd.DataFrame, ScalingParams]:
    """
    Divide order flows by their training-set standard deviation.
    Returns scaled DataFrame and the scaling parameters for later inversion.
    """
    train = of.iloc[:train_end_idx]
    params = ScalingParams(of_std=train.std())
    scaled = of / params.of_std
    return scaled, params


def build_supervised(
    fx: pd.DataFrame,
    of: pd.DataFrame,
    currency: str,
    segment: str,
    lags: int = 8,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Build (X, y) arrays for a given currency–segment pair.

    X shape : (n_samples, lags, n_features)
              features = [OF_am, OF_hf, OF_co, OF_pc, FX_return]
              (all four segments always included as exogenous inputs)
    y shape : (n_samples,)  — one-step-ahead FX log return
    """
    # Collect all four order-flow segments for this currency
    seg_cols = [f"{currency}_{s}" for s in
                ["asset_managers", "hedge_funds", "corporates", "private_clients"]]
    features = pd.concat([of[seg_cols], fx[[currency]]], axis=1).dropna()

    X_rows, y_rows, dates = [], [], []
    for i in range(lags, len(features) - 1):
        X_rows.append(features.iloc[i - lags: i].values)   # (lags, n_features)
        y_rows.append(features.iloc[i + 1][currency])       # next period return
        dates.append(features.index[i + 1])

    X = np.array(X_rows, dtype=np.float32)   # (n, lags, 5)
    y = np.array(y_rows, dtype=np.float32)   # (n,)
    return X, y, pd.DatetimeIndex(dates)
