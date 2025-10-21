# TODO(cursor): add β-neutral and sector-neutral constraints using factor loadings; promote L1 turnover with auxiliary variables.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import cvxpy as cp
import scipy.sparse as sp
import osqp
import yaml


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


def qp_optimize(
    expected: pd.Series,
    prev_weights: Optional[pd.Series] = None,
    returns_window: Optional[pd.DataFrame] = None,
    constraints_yaml: str = "constraints.yaml",
    exposures: Optional[pd.DataFrame] = None,
    turnover_gamma: Optional[float] = None,
) -> pd.Series:
    cfg = yaml.safe_load(open(constraints_yaml)) if constraints_yaml and len(constraints_yaml) else {}
    names = expected.index
    n = len(names)

    if turnover_gamma is None:
        turnover_gamma = float(cfg.get("turnover_gamma", 0.0))
    enforce_beta = bool(cfg.get("enforce_beta_neutral", False) or cfg.get("beta_neutral", False))
    enforce_sector = bool(cfg.get("enforce_sector_neutral", False) or cfg.get("sector_neutral", False))
    beta_target = float(cfg.get("beta_target", 0.0))

    # Risk (P)
    P = np.eye(n)
    if returns_window is not None and returns_window.shape[0] > 10:
        from .risk import shrink_cov

        cov = shrink_cov(returns_window)
        P = cov.reindex(index=names, columns=names).fillna(0.0).values
    P = 2.0 * P

    r = expected.values.astype(float)
    q = -r

    # Core constraints config
    G = float(cfg.get("gross_limit", 2.0))
    net_target = float(cfg.get("net_target", 0.0))
    cap = float(cfg.get("position_cap_bps", 100)) / 10000.0

    # Decision vector
    use_turnover = prev_weights is not None and turnover_gamma > 0
    if use_turnover:
        P_big = sp.block_diag([sp.csc_matrix(P), sp.csc_matrix((n, n))], format="csc")
        q_big = np.concatenate([q, np.full(n, turnover_gamma, dtype=float)])
    else:
        P_big = sp.csc_matrix(P)
        q_big = q

    rows = []
    l_list, u_list = [], []

    Iw = sp.eye(n, format="csr")

    # Net sum(w) = net_target
    net_row = sp.hstack([sp.csr_matrix(np.ones((1, n))), sp.csr_matrix((1, n))]) if use_turnover else sp.csr_matrix(np.ones((1, n)))
    rows.append(net_row); l_list.append(net_target); u_list.append(net_target)

    # Box bounds
    row_up = sp.hstack([Iw, sp.csr_matrix((n, n))]) if use_turnover else Iw
    row_lo = -row_up
    rows.extend([row_up, row_lo])
    l_list.extend([-np.inf] * n + [-np.inf] * n)
    u_list.extend([cap] * n + [cap] * n)

    # Factor neutrality
    if exposures is not None and exposures.size > 0:
        E_df = exposures.copy()
        # Filter rows by flags
        if not enforce_beta and "MKT" in E_df.index:
            E_df = E_df.drop(index="MKT")
        if not enforce_sector:
            # keep only MKT if requested, else none
            if enforce_beta and "MKT" in exposures.index:
                E_df = exposures.loc[["MKT"]]
            else:
                E_df = E_df.loc[E_df.index.intersection(["MKT"]) * 0]  # empty

        if E_df.shape[0] > 0:
            E = sp.csr_matrix(E_df.reindex(columns=names).fillna(0.0).values)
            row_E = sp.hstack([E, sp.csr_matrix((E.shape[0], n))]) if use_turnover else E
            rows.append(row_E)
            targets = [beta_target if idx == "MKT" else 0.0 for idx in E_df.index]
            l_list.extend(targets); u_list.extend(targets)

    # Turnover L1 via auxiliary variables
    if use_turnover:
        wp = prev_weights.reindex(names).fillna(0.0).values
        A1 = sp.hstack([Iw, -Iw])
        A2 = sp.hstack([-Iw, -Iw])
        rows.extend([A1, A2])
        l_list.extend([-np.inf] * n + [-np.inf] * n)
        u_list.extend(list(wp) + list(-wp))
        A3 = sp.hstack([sp.csr_matrix((n, n)), Iw])
        rows.append(A3); l_list.extend([0.0] * n); u_list.extend([np.inf] * n)

    A = sp.vstack(rows).tocsc()
    l = np.asarray(l_list, dtype=float)
    u = np.asarray(u_list, dtype=float)

    prob = osqp.OSQP()
    prob.setup(P=P_big, q=q_big, A=A, l=l, u=u, verbose=False)
    res = prob.solve()
    sol = res.x[:n] if use_turnover else res.x
    w = pd.Series(sol, index=names).clip(lower=-cap, upper=cap)

    gross = w.abs().sum()
    if gross > G:
        w *= G / (gross + 1e-12)
    return w


