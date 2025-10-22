# TODO(cursor): export weights_{YYYYMM}.csv & returns_{YYYYMM}.csv; add attribution by factor/sector.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from pathlib import Path
import json

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
        output_dir: Optional[str] = None,
        exposures_provider: Optional[callable] = None,
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
        
        # Track infeasible solves from optimizer metadata
        infeasible_days = 0

        # Optional monthly CSV exports
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            
            # Compute diagnostics for each day
            diagnostics = []
            for dt in strategy_ret.index:
                w_t = weights.loc[dt]
                # Track infeasibility flag from optimizer
                infeasible = bool(getattr(getattr(w_t, "attrs", {}), "get", lambda *_: False)("infeasible", False)) \
                             or bool(getattr(w_t, "attrs", {}).get("infeasible", False))
                if infeasible:
                    infeasible_days += 1
                net = float(w_t.sum())
                beta = np.nan
                sector_l2 = np.nan
                
                if exposures_provider is not None:
                    try:
                        # Get exposures for this date
                        expos = exposures_provider(dt, list(w_t.index), rets.loc[:dt].tail(60))
                        if "MKT" in expos.index:
                            beta = float((expos.loc["MKT", w_t.index] @ w_t.values))
                        # Sector L2 norm
                        E_sect = expos.drop(index="MKT", errors="ignore")
                        if E_sect.shape[0] > 0:
                            v = E_sect.loc[:, w_t.index] @ w_t.values
                            sector_l2 = float(np.sqrt(np.square(v).sum()))
                    except Exception:
                        pass
                
                diagnostics.append({
                    "ret": strategy_ret.loc[dt],
                    "net": net,
                    "beta": beta,
                    "sector_l2": sector_l2
                })
            
            diag_df = pd.DataFrame(diagnostics, index=strategy_ret.index)
            
            # Iterate per YYYYMM
            months = strategy_ret.index.to_series().dt.strftime("%Y%m")
            for ym, idxs in months.groupby(months).groups.items():
                # returns
                sr = strategy_ret.loc[idxs]
                sr.to_csv(out / f"returns_{ym}.csv", header=["ret"])
                # weights
                wdf = weights.loc[idxs]
                wdf.to_csv(out / f"weights_{ym}.csv")
                # diagnostics cutout
                diag_slice = diag_df.loc[idxs]
                diag_slice.to_csv(out / f"cutout_ret_data_{ym}.csv")
            
            # Persist minimal metrics alongside monthly CSVs
            meta = {
                "n_days": int(len(strategy_ret)),
                "infeasible_days": int(infeasible_days),
            }
            (out / "metrics.json").write_text(json.dumps(meta, indent=2))
        
        return strategy_ret, summary


