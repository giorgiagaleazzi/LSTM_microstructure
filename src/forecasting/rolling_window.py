"""
Expanding-window forecasting engine.

Every model in the repository uses this class.
"""

from __future__ import annotations

import numpy as np


class RollingForecast:

    """
    Recursive expanding-window forecasting.

    Parameters
    ----------
    model
        ForecastModel

    initial_window
        Initial estimation sample.
    """

    def __init__(
        self,
        model,
        initial_window: int,
    ):

        self.model = model

        self.initial_window = initial_window

    def run(
        self,
        X,
        y,
    ):

        forecasts = []

        actual = []

        forecast_dates = []

        for t in range(
            self.initial_window,
            len(y),
        ):

            X_train = X[:t]

            y_train = y[:t]

            X_test = X[t : t + 1]

            self.model.fit(
                X_train,
                y_train,
            )

            pred = self.model.predict(
                X_test,
            )

            forecasts.append(
                float(pred[0])
            )

            actual.append(
                float(y[t])
            )

            forecast_dates.append(t)

        return {
            "forecast": np.asarray(forecasts),
            "actual": np.asarray(actual),
            "index": np.asarray(forecast_dates),
        }
