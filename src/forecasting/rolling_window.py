"""
Walk-forward (expanding window) validation engine — paper Section 3.5.

At each step t in the OOS period:
  1. Scale order flows using training-set std dev (up to t, not beyond).
  2. Build supervised dataset.
  3. Re-train model from scratch on all data up to t.
  4. Predict Δs_{t+1}.
  5. Record prediction; advance t by one week.

This produces strictly genuine OOS forecasts with no lookahead bias.
~107 OOS predictions for the 2005-11-2007-11 window.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import pandas as pd


@dataclass
class WalkForwardResult:
    dates:       pd.DatetimeIndex
    actuals:     np.ndarray
    predictions: np.ndarray
    currency:    str
    segment:     str
    model_name:  str


def walk_forward(
    fx:          pd.DataFrame,
    of:          pd.DataFrame,
    currency:    str,
    segment:     str,
    oos_start:   str | pd.Timestamp,
    lags:        int,
    build_X_y:   Callable,       # from preprocessing.build_supervised
    train_model: Callable,       # returns fitted model with .predict()
    cfg:         dict,
    model_name:  str = "lstm",
    seed:        int = 42,
    verbose:     bool = False,
) -> WalkForwardResult:
    """
    Expanding-window walk-forward.

    Parameters
    ----------
    build_X_y   : callable(fx, of_scaled, currency, segment, lags) -> (X, y, dates)
    train_model : callable(X_train, y_train, cfg, seed) -> fitted model
    """
    from src.data.preprocessing import scale_order_flow

    oos_start = pd.Timestamp(oos_start)
    all_dates = fx.index

    pred_list, actual_list, date_list = [], [], []

    # First training window: everything before oos_start
    t0 = (all_dates < oos_start).sum()

    for t in range(t0, len(all_dates) - 1):
        if verbose and (t - t0) % 10 == 0:
            print(f"  {model_name} | {currency} | {segment} | step {t - t0 + 1}")

        # --- scale using data up to t only (no lookahead) ---
        of_scaled, _ = scale_order_flow(of, train_end_idx=t)

        # --- build supervised dataset ---
        X, y, dates = build_X_y(fx, of_scaled, currency, segment, lags)

        # keep only rows whose forecast date <= all_dates[t+1]
        mask_train = dates <= all_dates[t]
        mask_pred  = dates == all_dates[t + 1]

        if mask_train.sum() < lags + 5 or mask_pred.sum() == 0:
            continue

        X_train, y_train = X[mask_train], y[mask_train]
        X_pred            = X[mask_pred]

        # --- train from scratch ---
        model = train_model(X_train, y_train, cfg, seed=seed + t)

        # --- predict ---
        yhat = model.predict(X_pred).ravel()[0]
        pred_list.append(yhat)
        actual_list.append(y[mask_pred][0])
        date_list.append(all_dates[t + 1])

    return WalkForwardResult(
        dates       = pd.DatetimeIndex(date_list),
        actuals     = np.array(actual_list),
        predictions = np.array(pred_list),
        currency    = currency,
        segment     = segment,
        model_name  = model_name,
    )
