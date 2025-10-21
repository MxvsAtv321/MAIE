# TODO(cursor): add β-neutral and sector-neutral constraints using factor loadings; promote L1 turnover with auxiliary variables.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import cvxpy as cp
import scipy.sparse as sp
import osqp


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

        # Use OSQP formulation for future L1 turnover extensions
        P = 2.0 * Sigma  # OSQP objective uses 0.5 x^T P x
        q = -mu

        # Base constraints
        G = self.gross_limit
        cap = self.weight_cap
        if self.long_only:
            # sum w = 1, 0 <= w <= cap
            A = sp.vstack([
                sp.csr_matrix(np.ones((1, n))),
                sp.eye(n),
                -sp.eye(n),
            ])
            u = np.concatenate([
                np.array([1.0]),
                np.full(n, cap),
                np.zeros(n),
            ])
            l = np.concatenate([
                np.array([1.0]),
                np.zeros(n),
                -np.full(n, np.inf),
            ])
        else:
            # sum w = 0, |w|_1 <= G, |w| <= cap
            # Implement |w|_1 <= G via box and post-scale; primary constraints: sum=0, |w|<=cap
            A = sp.vstack([
                sp.csr_matrix(np.ones((1, n))),
                sp.eye(n),
                -sp.eye(n),
            ])
            u = np.concatenate([
                np.array([0.0]),
                np.full(n, cap),
                np.full(n, cap),
            ])
            l = np.concatenate([
                np.array([0.0]),
                -np.full(n, np.inf),
                -np.full(n, np.inf),
            ])

        prob = osqp.OSQP()
        prob.setup(P=sp.csc_matrix(P), q=q, A=A.tocsc(), l=l, u=u, verbose=False)
        res = prob.solve()
        if res.x is None:
            raise RuntimeError("Optimization failed")
        w = pd.Series(res.x, index=assets).clip(lower=-cap, upper=cap)
        # Normalize gross exposure if needed
        gross = w.abs().sum()
        if not self.long_only and gross > G:
            w *= G / (gross + 1e-12)
        return OptimizeResult(weights=w, objective_value=float(res.info.obj_val))


