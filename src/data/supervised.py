"""
Convert time-series into supervised learning format.
"""

from __future__ import annotations

import numpy as np


class SupervisedDataset:

    """
    Transform

    x(t)

    into

    x(t-k)...x(t-1)->y(t)

    """

    def __init__(

        self,

        lags: int = 4

    ):

        self.lags = lags

    def transform(

        self,

        X,

        y

    ):

        X = np.asarray(X)

        y = np.asarray(y)

        xs = []

        ys = []

        for i in range(

            self.lags,

            len(X)

        ):

            xs.append(

                X[i-self.lags:i]

            )

            ys.append(

                y[i]

            )

        return (

            np.asarray(xs),

            np.asarray(ys)

        )
