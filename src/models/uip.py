"""
Uncovered Interest Parity.
"""

from __future__ import annotations

from sklearn.linear_model import LinearRegression

from .base import ForecastModel


class UIP(ForecastModel):

    def __init__(self):

        self.model = LinearRegression()

    def fit(

        self,

        X,

        y

    ):

        self.model.fit(

            X,

            y

        )

    def predict(

        self,

        X

    ):

        return self.model.predict(X)
