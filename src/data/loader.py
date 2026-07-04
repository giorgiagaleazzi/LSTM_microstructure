"""
Raw data loader.

Expected files in data/raw/
  - DataSourceEUR.xlsx  : weekly FX spot rates (Reuters/DataStream)
  - dt_chapter1.xls     : UBS customer order flow by segment

Column conventions after loading
  - FX returns   : log returns, one column per currency (AUD, CAD, …)
  - Order flow   : one column per (currency, segment) combination,
                   e.g. AUD_asset_managers, AUD_hedge_funds, …
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path


CURRENCIES = ["AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NOK", "NZD", "SEK"]
SEGMENTS   = ["asset_managers", "hedge_funds", "corporates", "private_clients"]


def load_fx(path: str | Path) -> pd.DataFrame:
    """
    Load weekly FX spot rates and compute log returns.
    Returns a DataFrame indexed by date with one column per currency.
    """
    df = pd.read_excel(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    # Compute log returns: r_t = ln(s_t) - ln(s_{t-1})
    returns = np.log(df).diff().dropna()
    returns.columns = [c.upper() for c in returns.columns]
    return returns


def load_order_flow(path: str | Path) -> pd.DataFrame:
    """
    Load UBS customer order flow disaggregated by segment.
    Returns a DataFrame indexed by date with columns:
      {CURRENCY}_{segment}
    Order flow is defined as buyer-initiated minus seller-initiated
    transactions (positive = net foreign currency purchase).
    """
    xl = pd.ExcelFile(path)
    frames = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet, index_col=0, parse_dates=True)
        df = df.sort_index()
        # Expect columns: asset_managers, hedge_funds, corporates, private_clients
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df = df.rename(columns={c: f"{sheet.upper()}_{c}" for c in df.columns})
        frames.append(df)
    combined = pd.concat(frames, axis=1)
    return combined


def load_all(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load FX returns and order flow; align on common dates and date range."""
    raw = Path(cfg["data"]["raw_dir"])
    fx = load_fx(raw / cfg["data"]["fx_file"])
    of = load_order_flow(raw / cfg["data"]["order_flow_file"])

    # Align
    idx = fx.index.intersection(of.index)
    start = pd.Timestamp(cfg["data"]["start_date"])
    end   = pd.Timestamp(cfg["data"]["end_date"])
    idx   = idx[(idx >= start) & (idx <= end)]

    return fx.loc[idx], of.loc[idx]
