from __future__ import annotations

from datetime import datetime
from typing import Iterable, List

import numpy as np
import pandas as pd


def generate_synthetic_prices(
    start: str = "2018-01-01",
    end: str = "2022-12-31",
    tickers: Iterable[str] | None = None,
    seed: int | None = 42,
    annual_vol: float = 0.20,
) -> pd.DataFrame:
    """
    Generate synthetic daily close prices for a small universe.

    Returns a DataFrame indexed by date with tickers as columns.
    """
    if tickers is None:
        tickers = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    tickers = list(tickers)

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    num_days = len(dates)
    num_assets = len(tickers)

    daily_vol = annual_vol / np.sqrt(252.0)
    drift = 0.05 / 252.0  # modest drift

    # Correlated Gaussian shocks
    base_corr = 0.2
    cov = (1 - base_corr) * np.eye(num_assets) + base_corr * np.ones((num_assets, num_assets))
    chol = np.linalg.cholesky(cov)

    shocks = rng.standard_normal((num_days, num_assets)) @ chol.T
    rets = drift + daily_vol * shocks

    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    df = pd.DataFrame(prices, index=dates, columns=tickers).astype("float64")
    return df


