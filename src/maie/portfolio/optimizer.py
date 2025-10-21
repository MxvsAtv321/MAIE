# TODO(cursor): add β-neutral and sector-neutral constraints using factor loadings; promote L1 turnover with auxiliary variables.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import cvxpy as cp


@dataclass
class OptimizeResult:
    weights: pd.Series
    objective_value: float


class MeanVarianceOptimizer:
    """
    Mean-variance optimizer with L2 risk using OSQP.

    Constraints:
    - sum(weights) = 0 (dollar-neutral) or = 1 (long-only) depending on `long_only` flag
    - gross exposure cap via |w|_1 <= gross_limit
    - individual weight caps [-w_cap, w_cap]

    TODO(cursor): add β-neutral and sector-neutral constraints using factor loadings; promote L1 turnover with auxiliary variables.
    """

    def __init__(
        self,
        risk_aversion: float = 10.0,
        gross_limit: float = 1.0,
        weight_cap: float = 0.10,
        long_only: bool = False,
    ) -> None:
        self.risk_aversion = risk_aversion
        self.gross_limit = gross_limit
        self.weight_cap = weight_cap
        self.long_only = long_only

    def optimize(self, alphas: pd.Series, cov: pd.DataFrame) -> OptimizeResult:
        assets = alphas.index.tolist()
        n = len(assets)
        mu = alphas.values.astype(float)
        Sigma = cov.loc[assets, assets].values.astype(float)

        w = cp.Variable(n)

        # Objective: maximize mu^T w - lambda * w^T Sigma w
        risk = cp.quad_form(w, Sigma)
        objective = cp.Maximize(mu @ w - self.risk_aversion * risk)

        constraints = []
        if self.long_only:
            constraints += [cp.sum(w) == 1.0, w >= 0.0, w <= self.weight_cap]
        else:
            constraints += [cp.sum(w) == 0.0, cp.norm1(w) <= self.gross_limit, cp.abs(w) <= self.weight_cap]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        if w.value is None:
            raise RuntimeError("Optimization failed")
        weights = pd.Series(w.value, index=assets)
        return OptimizeResult(weights=weights, objective_value=float(prob.value))


