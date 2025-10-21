#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd

from maie.data.synthetic import generate_synthetic_prices
from maie.features.tabular import build_features
from maie.models.structured import StructuredModel
from maie.backtest.engine import BacktestEngine
from maie.portfolio.optimizer import qp_optimize
from maie.portfolio.exposures import sector_one_hot, beta_exposures


def exposures_provider(dt, tickers, returns_window: pd.DataFrame):
    sect = sector_one_hot(tickers)
    betas = beta_exposures(returns_window.reindex(columns=tickers))
    E = pd.concat([sect, betas.to_frame().T], axis=0)
    return E


def main() -> None:
    tickers = [f"SIM{i:04d}" for i in range(200)]
    prices = generate_synthetic_prices(end="2024-12-31", tickers=tickers)
    X, y = build_features(prices)
    model = StructuredModel()
    model.fit(X, y)

    # Expected returns proxy: last-day predictions per asset
    last_day = X.index.get_level_values(0).max()
    alphas = model.predict(X.loc[last_day])

    # Build rolling cov from returns
    rets = prices.pct_change().dropna()
    cov_last = rets.rolling(60).cov().dropna().loc[rets.index[-1]]

    # One step optimization on last day with constraints
    w = qp_optimize(
        expected=alphas,
        prev_weights=None,
        returns_window=rets.tail(60).reindex(columns=alphas.index),
        constraints_yaml="constraints.yaml",
        exposures=exposures_provider(last_day, list(alphas.index), rets.tail(60)),
    )

    # Hold from last_day onward and backtest
    weights_df = pd.DataFrame(index=prices.index, columns=alphas.index, data=0.0)
    weights_df.loc[last_day:, :] = w.values
    engine = BacktestEngine(transaction_cost_bps=5.0)
    strat_ret, summary = engine.run(prices, weights_df, output_dir="outputs_constraints", exposures_provider=exposures_provider)
    print("==== Constrained backtest summary ====")
    print(summary)

    # Sanity: neutrality
    E = exposures_provider(last_day, list(w.index), rets.tail(60))
    print("Net:", float(w.sum()))
    if "MKT" in E.index:
        print("Beta approx:", float(E.loc["MKT", w.index] @ w.values))
    sect = E.drop(index="MKT", errors="ignore")
    if len(sect) > 0:
        se = (sect @ w.values)
        print("Sector exposure L2:", float((se**2).sum() ** 0.5))


if __name__ == "__main__":
    main()


