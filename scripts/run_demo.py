from __future__ import annotations

import numpy as np
import pandas as pd

from maie.data import generate_synthetic_prices
from maie.features import build_features
from maie.models import StructuredModel
from maie.portfolio import MeanVarianceOptimizer
from maie.backtest import BacktestEngine


def main() -> None:
    prices = generate_synthetic_prices()
    X, y = build_features(prices)

    model = StructuredModel()
    result = model.fit(X, y)
    print(f"Model OOS R2: {result.oos_r2:.4f}")

    # Build last-day alphas per asset
    last_day = X.index.get_level_values(0).max()
    X_last = X.loc[last_day]
    alphas = model.predict(X_last)

    # Simple covariance from recent returns
    rets = prices.pct_change().dropna()
    cov = rets.rolling(60).cov().dropna()
    cov_last = cov.loc[cov.index.get_level_values(0).max()]

    optimizer = MeanVarianceOptimizer(risk_aversion=5.0, gross_limit=1.0, weight_cap=0.2, long_only=False)
    opt = optimizer.optimize(alphas, cov_last)
    weights_last = opt.weights

    # Hold weights constant for simplicity
    weights_df = pd.DataFrame(index=prices.index, columns=weights_last.index, data=0.0)
    weights_df.loc[last_day:, :] = weights_last.values

    engine = BacktestEngine(transaction_cost_bps=5.0)
    strat_ret, summary = engine.run(prices, weights_df, spread_bps=5.0)

    print("Demo Summary:")
    print(f"  Sharpe: {summary.sharpe:.2f}")
    print(f"  CAGR:   {summary.cagr:.2%}")
    print(f"  MaxDD:  {summary.max_drawdown:.2%}")


if __name__ == "__main__":
    main()


