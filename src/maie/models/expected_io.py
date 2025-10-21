from __future__ import annotations
from pathlib import Path
import pandas as pd

def load_expected_dir(path: str | Path) -> pd.DataFrame:
    """
    Loads monthly `expected_YYYYMM.parquet` files and returns a single
    wide DataFrame aligned by date (index) and tickers (columns).
    """
    p = Path(path)
    parts = sorted(p.glob("expected_*.parquet"))
    frames = []
    for f in parts:
        df = pd.read_parquet(f)
        df.index = pd.to_datetime(df.index)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No expected_*.parquet in {p}")
    expected = pd.concat(frames, axis=0).sort_index()
    # de-duplicate if overlapping months
    expected = expected[~expected.index.duplicated(keep="last")]
    return expected

def load_expected_latest(path: str | Path) -> pd.DataFrame:
    """Loads `expected_latest.parquet` (1-row snapshot)."""
    p = Path(path) / "expected_latest.parquet" if Path(path).is_dir() else Path(path)
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    return df
