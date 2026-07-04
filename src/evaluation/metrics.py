"""
Forecast accuracy metrics — paper Section 3.5 (RMSE, MAPE).
"""
from __future__ import annotations
import numpy as np


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def rmsfe_ratio(
    actual: np.ndarray,
    predicted: np.ndarray,
    benchmark: np.ndarray,
) -> float:
    """RMSFE of model relative to benchmark (< 1 = outperforms)."""
    return rmse(actual, predicted) / rmse(actual, benchmark)
