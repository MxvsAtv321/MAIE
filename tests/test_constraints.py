from __future__ import annotations
import numpy as np
import pandas as pd
from maie.data.synthetic import generate_synthetic_prices
from maie.features.tabular import build_features
from maie.models.structured import StructuredModel
from maie.portfolio.optimizer import qp_optimize
from maie.portfolio.exposures import sector_one_hot, beta_exposures

def _exposures(dt, tickers, window: pd.DataFrame) -> pd.DataFrame:
    sect = sector_one_hot(tickers)
    bet = beta_exposures(window.reindex(columns=tickers))
    return pd.concat([sect, bet.to_frame().T], axis=0)

def test_neutrality_and_turnover():
    tickers = [f"SIM{i:03d}" for i in range(40)]
    close = generate_synthetic_prices(tickers=tickers, end="2024-12-31", seed=99)
    rets = close.pct_change()
    X, y = build_features(close)
    model = StructuredModel()
    result = model.fit(X, y)
    # Get last day predictions
    last_day = X.index.get_level_values(0).max()
    X_last = X.loc[last_day]
    exp = model.predict(X_last).reindex(tickers).dropna()

    prev_w = pd.Series(0.0, index=exp.index)  # start from flat; turnover defined vs. 0
    window = rets.tail(60).reindex(columns=exp.index)
    E = _exposures(close.index[-1], list(exp.index), window)
    w = qp_optimize(
        expected=exp,
        prev_weights=prev_w,
        returns_window=window,
        constraints_yaml="constraints.yaml",
        exposures=E,
    )

    # Read tolerances from file
    import yaml
    cfg = yaml.safe_load(open("constraints.yaml"))
    beta_tol = float(cfg.get("beta_tolerance", 0.0)) + 2e-4
    sect_tol = float(cfg.get("sector_tolerance", 0.0)) + 2e-4

    # Beta near target within tolerance
    if "MKT" in E.index:
        beta_target = float(cfg.get("beta_target", 0.0))
        beta_exp = float((E.loc["MKT", w.index] @ w.values))
        assert abs(beta_exp - beta_target) <= beta_tol

    # Each sector within tolerance band
    for ridx in E.index:
        if ridx == "MKT":
            continue
        val = float((E.loc[ridx, w.index] @ w.values))
        assert abs(val) <= sect_tol

    # Turnover penalty: from zero prev_w, the L1 equals gross
    gross = float(w.abs().sum())
    assert gross <= float(cfg.get("gross_limit", 2.0)) + 1e-9
