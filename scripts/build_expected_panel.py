#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import pandas as pd
import time, json, os
from maie.data.synthetic import generate_synthetic_prices
from maie.models.rolling import build_expected_panel_from_prices, RollingTrainerCfg

OUTDIR = Path("expected"); OUTDIR.mkdir(exist_ok=True, parents=True)

def main() -> None:
    t0 = time.perf_counter()
    tickers = [f"SIM{i:04d}" for i in range(800)]
    close = generate_synthetic_prices(tickers=tickers, end="2024-12-31", seed=21)
    cfg = RollingTrainerCfg(horizon=5, train_window_days=504, cv_folds=3, step="M")
    expected = build_expected_panel_from_prices(close, cfg)
    # Persist monthly partitions
    for ym, chunk in expected.groupby(pd.Grouper(freq="M")):
        if pd.isna(ym): 
            continue
        ymstr = ym.strftime("%Y%m")
        (OUTDIR / f"expected_{ymstr}.parquet").write_bytes(chunk.to_parquet())
    # Latest snapshot for the API or research
    (OUTDIR / "expected_latest.parquet").write_bytes(expected.tail(1).to_parquet())
    build_seconds = time.perf_counter() - t0
    # Collect file stats
    files = sorted([p for p in OUTDIR.glob("expected_*.parquet")])
    total_bytes = sum(p.stat().st_size for p in files) + (OUTDIR / "expected_latest.parquet").stat().st_size
    
    # Get unique dates for coherence check
    idx = expected.index.unique().sort_values()
    meta = {
        "shape": [int(expected.shape[0]), int(expected.shape[1])],
        "n_unique_dates": int(len(idx)),
        "head_dates": [str(d.date()) for d in idx[:3]],
        "tail_dates": [str(d.date()) for d in idx[-3:]],
        "start": str(idx.min().date()) if len(idx) else None,
        "end": str(idx.max().date()) if len(idx) else None,
        "n_files": int(len(files) + 1),
        "total_bytes": int(total_bytes),
        "build_seconds": float(build_seconds),
    }
    (OUTDIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote expected panel with shape {expected.shape} to {OUTDIR}/ in {build_seconds:.2f}s")

if __name__ == "__main__":
    main()
