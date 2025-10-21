#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import pandas as pd
from maie.data.synthetic import generate_synthetic_prices
from maie.models.rolling import build_expected_panel_from_prices, RollingTrainerCfg

OUTDIR = Path("expected"); OUTDIR.mkdir(exist_ok=True, parents=True)

def main() -> None:
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
    print(f"Wrote expected panel with shape {expected.shape} to {OUTDIR}/")

if __name__ == "__main__":
    main()
