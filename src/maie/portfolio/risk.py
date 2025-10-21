from __future__ import annotations

import numpy as np
import pandas as pd


def shrink_cov(returns_window: pd.DataFrame, ridge: float = 1e-6) -> pd.DataFrame:
    """Simple covariance shrinkage: empirical covariance + ridge on diagonal.

    Returns a DataFrame aligned to asset columns.
    """
    cols = list(returns_window.columns)
    if len(returns_window) < 2:
        return pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    cov = returns_window.cov().reindex(index=cols, columns=cols).fillna(0.0)
    cov.values[range(len(cols)), range(len(cols))] += ridge
    return cov


