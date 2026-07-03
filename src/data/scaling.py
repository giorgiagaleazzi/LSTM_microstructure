"""
Scaling methods.
"""

from __future__ import annotations

import pandas as pd

from sklearn.preprocessing import MinMaxScaler


class Scaler:

    """
    Wrapper around sklearn scaler.

    The scaler is fitted ONLY on training data.
    """

    def __init__(self):

        self.scaler = MinMaxScaler(
            feature_range=(-1, 1)
        )

    def fit(self, train: pd.DataFrame):

        self.scaler.fit(train)

    def transform(self, data: pd.DataFrame):

        values = self.scaler.transform(data)

        return pd.DataFrame(
            values,
            columns=data.columns,
            index=data.index
        )

    def fit_transform(self, train):

        values = self.scaler.fit_transform(train)

        return pd.DataFrame(
            values,
            columns=train.columns,
            index=train.index
        )

    def inverse(self, values):

        return self.scaler.inverse_transform(values)
