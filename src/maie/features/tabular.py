from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def _safe_pct_change(prices: pd.DataFrame, periods: int) -> pd.DataFrame:
    rets = prices.pct_change(periods=periods)
    return rets.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_features(prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Baseline tabular features:
    - momentum (1M, 3M, 6M)
    - realized volatility (20d)
    - short-term reversal (5d)

    Returns (X, y) where X is a MultiIndex DataFrame with columns per feature
    and y is next-day return as target.
    """
    rets1 = _safe_pct_change(prices, 1)
    rets5 = _safe_pct_change(prices, 5)
    rets20 = _safe_pct_change(prices, 20)
    rets63 = _safe_pct_change(prices, 63)
    rets126 = _safe_pct_change(prices, 126)

    # Features per asset
    mom_1m = rets20
    mom_3m = rets63
    mom_6m = rets126
    vol_20 = rets1.rolling(20).std().fillna(0.0)
    rev_5 = -rets5

    features = {
        "mom_1m": mom_1m,
        "mom_3m": mom_3m,
        "mom_6m": mom_6m,
        "vol_20": vol_20,
        "rev_5": rev_5,
    }

    # Stack into long-form table with MultiIndex (date, asset)
    X_parts = []
    for name, df in features.items():
        part = df.stack().rename(name)
        X_parts.append(part)
    X = pd.concat(X_parts, axis=1)

    # Target: next-day return
    next_ret = rets1.shift(-1)
    y = next_ret.stack().rename("target")

    # Align indices
    X, y = X.align(y, join="inner", axis=0)
    return X.astype("float64"), y.astype("float64")


