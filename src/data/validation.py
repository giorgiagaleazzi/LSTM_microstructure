"""
Dataset validation.
"""

import pandas as pd


class DatasetValidator:

    """
    Validate datasets before training.
    """

    @staticmethod
    def no_missing(df: pd.DataFrame):

        if df.isna().sum().sum() != 0:

            raise ValueError(
                "Dataset contains missing values."
            )

    @staticmethod
    def numeric(df: pd.DataFrame):

        bad = []

        for c in df.columns:

            if not pd.api.types.is_numeric_dtype(df[c]):

                bad.append(c)

        if bad:

            raise ValueError(

                f"Non numeric columns: {bad}"

            )

    @staticmethod
    def sufficient_rows(

        df,

        minimum=100

    ):

        if len(df) < minimum:

            raise ValueError(

                "Dataset too small."

            )
