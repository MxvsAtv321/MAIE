from __future__ import annotations

import pandas as pd


def sector_one_hot(tickers: list[str], sector_map: dict[str, str]) -> pd.DataFrame:
    """Rows=factors (sectors), cols=tickers; 1 if ticker in sector else 0."""
    sectors = sorted(set(sector_map.get(t, "Unknown") for t in tickers))
    mat = pd.DataFrame(0.0, index=sectors, columns=tickers)
    for t in tickers:
        mat.loc[sector_map.get(t, "Unknown"), t] = 1.0
    return mat


def beta_row(betas: pd.Series) -> pd.DataFrame:
    """Single-row DataFrame named 'MKT' for market beta neutrality."""
    return pd.DataFrame([betas.reindex(betas.index)], index=["MKT"])


