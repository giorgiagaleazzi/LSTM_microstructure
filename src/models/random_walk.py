"""
Random Walk benchmark.

Forecast

S(t+1)=S(t)
"""

from __future__ import annotations

import numpy as np

from .base import ForecastModel


class RandomWalk(ForecastModel):

    def fit(self, X, y):

        self.last = y[-1]

    def predict(self, X):

        return np.repeat(

            self.last,

            len(X)

        )
