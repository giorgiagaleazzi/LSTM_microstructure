"""
Forecast accuracy metrics.
"""

from __future__ import annotations

import numpy as np


def mse(

    actual,

    predicted,

):

    return np.mean(

        (

            actual - predicted

        )

        ** 2

    )


def rmse(

    actual,

    predicted,

):

    return np.sqrt(

        mse(

            actual,

            predicted,

        )

    )


def mae(

    actual,

    predicted,

):

    return np.mean(

        np.abs(

            actual

            - predicted

        )

    )


def mape(

    actual,

    predicted,

):

    actual = np.asarray(actual)

    predicted = np.asarray(predicted)

    return (

        np.mean(

            np.abs(

                (

                    actual

                    - predicted

                )

                / actual

            )

        )

        * 100

    )


def rmsfe(

    actual,

    predicted,

):

    return np.sqrt(

        np.mean(

            (

                actual

                - predicted

            )

            ** 2

        )

    )


def directional_accuracy(

    actual,

    predicted,

):

    return np.mean(

        np.sign(actual)

        == np.sign(predicted)

    )
