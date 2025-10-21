from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd


GICS_11 = [
    "Energy",
    "Materials",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Health Care",
    "Financials",
    "Information Technology",
    "Communication Services",
    "Utilities",
    "Real Estate",
]


def _deterministic_sector(ticker: str) -> str:
    h = int(hashlib.sha1(ticker.encode()).hexdigest(), 16)
    return GICS_11[h % len(GICS_11)]


def sector_one_hot(tickers: list[str], sector_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Rows=sectors, Cols=tickers; 1 if ticker belongs to sector else 0.

    If sector_map is None, uses a deterministic hash-based mapping for synthetic tickers.
    """
    if sector_map is None:
        sector_map = {t: _deterministic_sector(t) for t in tickers}
    sectors = sorted(set(sector_map.get(t, "Unknown") for t in tickers))
    M = pd.DataFrame(0.0, index=sectors, columns=tickers)
    for t in tickers:
        M.loc[sector_map.get(t, "Unknown"), t] = 1.0
    return M


def beta_exposures(returns_window: pd.DataFrame) -> pd.Series:
    """Single-row 'MKT' beta per ticker using equal-weight market proxy.

    returns_window: index=dates, columns=tickers
    """
    if returns_window.shape[0] < 2:
        return pd.Series(0.0, index=returns_window.columns, name="MKT")
    mkt = returns_window.mean(axis=1).fillna(0.0)
    var_m = float(mkt.var(ddof=1) or 1e-12)
    betas: dict[str, float] = {}
    for t in returns_window.columns:
        ri = returns_window[t].fillna(0.0)
        if len(ri) < 2:
            betas[t] = 0.0
            continue
        cov = float(np.cov(ri, mkt, ddof=1)[0, 1])
        betas[t] = cov / var_m if var_m > 0 else 0.0
    return pd.Series(betas, name="MKT")


