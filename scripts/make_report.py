#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import pandas as pd

def concat_monthlies(folder: Path, stem: str) -> pd.DataFrame:
    files = sorted(folder.glob(f"{stem}_*.csv"))
    frames = []
    for f in files:
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    out.index.name = "date"
    return out

def main() -> None:
    # Prefer outputs_from_expected/ if present; else fall back to outputs/
    base = Path("outputs_from_expected")
    if not base.exists():
        base = Path("outputs")
    returns = concat_monthlies(base, "returns")
    cutout  = concat_monthlies(base, "cutout_ret_data")
    if returns.empty:
        print(f"No monthly outputs in {base}/")
        return
    # Handle column overlap by using suffixes
    report = returns.join(cutout, how="left", rsuffix="_diag")
    out = base / "report_all.csv"
    report.to_csv(out)
    print(f"Report written to {out} (rows={len(report)})")

if __name__ == "__main__":
    main()
