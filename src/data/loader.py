"""
Load raw datasets used throughout the project.

The loader supports

- xlsx
- xls

and automatically validates the contents.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SUPPORTED = [".xlsx", ".xls"]


class DatasetLoader:

    """
    Load Excel datasets.
    """

    def __init__(self, path: str | Path):

        self.path = Path(path)

        if not self.path.exists():

            raise FileNotFoundError(self.path)

        if self.path.suffix.lower() not in SUPPORTED:

            raise ValueError(
                f"Unsupported file type {self.path.suffix}"
            )

    def load(self) -> pd.DataFrame:

        if self.path.suffix == ".xlsx":

            df = pd.read_excel(
                self.path,
                engine="openpyxl"
            )

        else:

            df = pd.read_excel(
                self.path,
                engine="xlrd"
            )

        return df

    @staticmethod
    def clean_columns(df: pd.DataFrame) -> pd.DataFrame:

        df.columns = (
            df.columns
            .str.strip()
            .str.replace(" ", "_")
            .str.lower()
        )

        return df

    @staticmethod
    def print_summary(df: pd.DataFrame):

        print("=" * 60)

        print("Rows :", len(df))

        print("Columns :", len(df.columns))

        print()

        print(df.columns.tolist())

        print("=" * 60)
