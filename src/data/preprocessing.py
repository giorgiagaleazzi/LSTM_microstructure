"""
Preprocessing utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class Preprocessor:

    """
    Data preprocessing.
    """

    @staticmethod
    def log_returns(series: pd.Series) -> pd.Series:
        """
        Compute

        r_t = log(S_t)-log(S_{t-1})

        """

        return np.log(series).diff()

    @staticmethod
    def difference(series: pd.Series) -> pd.Series:

        return series.diff()

    @staticmethod
    def standardise_order_flow(series: pd.Series):

        return series / series.std(ddof=1)

    @staticmethod
    def remove_missing(df: pd.DataFrame):

        return df.dropna().reset_index(drop=True)

    @staticmethod
    def chronological_split(
        df: pd.DataFrame,
        train_size: float = 0.70
    ):

        n = len(df)

        split = int(train_size * n)

        train = df.iloc[:split].copy()

        test = df.iloc[split:].copy()

        return train, test
