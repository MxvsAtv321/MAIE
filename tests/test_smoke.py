from __future__ import annotations

import pandas as pd

from maie.data import generate_synthetic_prices
from maie.features import build_features
from maie.models import StructuredModel
from maie.portfolio import MeanVarianceOptimizer
from maie.backtest import BacktestEngine


def test_end_to_end_smoke() -> None:
    prices = generate_synthetic_prices(end="2019-12-31", tickers=["A", "B", "C"])  # small for CI
    X, y = build_features(prices)
    assert len(X) > 100
    assert len(X) == len(y)

    model = StructuredModel()
    res = model.fit(X, y)
    assert res.model is not None

    last_day = X.index.get_level_values(0).max()
    X_last = X.loc[last_day]
    alphas = model.predict(X_last)
    assert alphas.shape[0] == len(prices.columns)

    rets = prices.pct_change().dropna()
    cov_last = rets.rolling(60).cov().dropna().iloc[-len(prices.columns):]
    cov_last = cov_last.reset_index(level=0, drop=True)

    optimizer = MeanVarianceOptimizer(risk_aversion=5.0, gross_limit=1.0, weight_cap=0.5)
    opt = optimizer.optimize(alphas, cov_last)
    weights = opt.weights
    assert abs(weights.sum()) < 1e-6  # dollar-neutral

    weights_df = pd.DataFrame(index=prices.index, columns=weights.index, data=0.0)
    weights_df.loc[last_day:, :] = weights.values

    engine = BacktestEngine(transaction_cost_bps=5.0)
    strat_ret, summary = engine.run(prices, weights_df)
    assert strat_ret.notna().all()
    assert isinstance(summary.sharpe, float)


