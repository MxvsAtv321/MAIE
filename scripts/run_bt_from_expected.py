#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import pandas as pd
from maie.data.synthetic import generate_synthetic_prices
from maie.backtest.engine import BacktestEngine
from maie.portfolio.optimizer import qp_optimize
from maie.portfolio.exposures import sector_one_hot, beta_exposures
from maie.models.expected_io import load_expected_dir

OUT = Path("outputs_from_expected")

def exposures_provider(dt, tickers, returns_window: pd.DataFrame):
    sect = sector_one_hot(tickers)
    bet = beta_exposures(returns_window.reindex(columns=tickers))
    return pd.concat([sect, bet.to_frame().T], axis=0)

def main() -> None:
    # 1) Load expected returns panel from parquet
    expected = load_expected_dir("expected")  # produced by build_expected_panel.py

    # 2) Load prices (synthetic placeholder; swap to real loader later)
    tickers = expected.columns.tolist()
    # Use similar length window for prices to avoid alignment issues
    n_days = max(400, len(expected) + 5)
    px = generate_synthetic_prices(tickers=tickers, end="2024-12-31", seed=33)
    # Align price index to expected
    px = px.reindex(expected.index).ffill().bfill()

    # 3) Run constrained backtest (writes monthly CSVs with diagnostics)
    # For now, use a simple approach: optimize on last day and hold
    last_day = expected.index[-1]
    alphas = expected.loc[last_day].dropna()
    
    # Get returns for optimization
    rets = px.pct_change()
    window = rets.tail(60).reindex(columns=alphas.index)
    E = exposures_provider(last_day, list(alphas.index), window)
    
    w = qp_optimize(
        expected=alphas,
        prev_weights=None,
        returns_window=window,
        constraints_yaml="constraints.yaml",
        exposures=E,
    )
    
    # Hold weights from last day onward
    weights_df = pd.DataFrame(index=px.index, columns=alphas.index, data=0.0)
    weights_df.loc[last_day:, :] = w.values
    
    engine = BacktestEngine(transaction_cost_bps=5.0)
    strat_ret, summary = engine.run(
        px, 
        weights_df, 
        output_dir=str(OUT), 
        exposures_provider=exposures_provider
    )
    
    print("==== Backtest (from expected panel) summary ====")
    print(summary)
    print(f"Wrote monthly weights/returns/cutouts to {OUT}/")

if __name__ == "__main__":
    main()
