"""
Portfolio performance measures.
"""

from __future__ import annotations

import numpy as np


def sharpe_ratio(

    returns,

    risk_free=0,

):

    excess = (

        returns

        - risk_free

    )

    return (

        np.mean(excess)

        / np.std(

            excess,

            ddof=1,

        )

    )


def downside_deviation(

    returns,

):

    negative = returns[

        returns < 0

    ]

    if len(negative) == 0:

        return 0

    return np.std(

        negative,

        ddof=1,

    )


def sortino_ratio(

    returns,

    risk_free=0,

):

    downside = downside_deviation(

        returns

    )

    if downside == 0:

        return np.nan

    return (

        np.mean(

            returns

            - risk_free

        )

        / downside

    )
