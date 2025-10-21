# TODO(cursor): export weights_{YYYYMM}.csv & returns_{YYYYMM}.csv; add attribution by factor/sector.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class BacktestSummary:
    sharpe: float
    cagr: float
    max_drawdown: float


class BacktestEngine:
    """
    Simple daily backtest: trade at close with next-day returns and transaction costs.

    TODO(cursor): export weights_{YYYYMM}.csv & returns_{YYYYMM}.csv; add attribution by factor/sector.
    """

    def __init__(self, transaction_cost_bps: float = 5.0) -> None:
        self.transaction_cost = transaction_cost_bps / 10000.0

    @staticmethod
    def _compute_stats(strategy_returns: pd.Series) -> BacktestSummary:
        mu = strategy_returns.mean() * 252.0
        sigma = strategy_returns.std(ddof=0) * np.sqrt(252.0)
        sharpe = 0.0 if sigma == 0 else mu / sigma

        cum = (1.0 + strategy_returns).cumprod()
        roll_max = cum.cummax()
        dd = cum / roll_max - 1.0
        max_dd = float(dd.min())

        n_years = max(1.0, len(strategy_returns) / 252.0)
        cagr = float(cum.iloc[-1] ** (1.0 / n_years) - 1.0)
        return BacktestSummary(sharpe=float(sharpe), cagr=cagr, max_drawdown=max_dd)

    def run(
        self,
        prices: pd.DataFrame,
        weights: pd.DataFrame,
        spread_bps: float = 5.0,
    ) -> Tuple[pd.Series, BacktestSummary]:
        rets = prices.pct_change().fillna(0.0)
        # Align
        weights = weights.reindex(index=rets.index).fillna(0.0)

        # Daily portfolio returns (next-day)
        daily_ret = (weights.shift().fillna(0.0) * rets).sum(axis=1)

        # Transaction cost on turnover
        turnover = (weights.diff().abs()).sum(axis=1)
        tcost = (spread_bps / 10000.0) * turnover
        strategy_ret = daily_ret - tcost

        summary = self._compute_stats(strategy_ret)
        return strategy_ret, summary


