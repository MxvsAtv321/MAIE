from __future__ import annotations
from pathlib import Path
from maie.data.synthetic import generate_synthetic_prices
from maie.models.rolling import build_expected_panel_from_prices, RollingTrainerCfg

def test_expected_panel_builds():
    close = generate_synthetic_prices(tickers=[f"SIM{i:03d}" for i in range(60)], end="2024-12-31", seed=5)
    cfg = RollingTrainerCfg(horizon=5, train_window_days=126, cv_folds=3, step="M")
    expected = build_expected_panel_from_prices(close, cfg)
    assert expected.shape[0] > 0 and expected.shape[1] > 0
