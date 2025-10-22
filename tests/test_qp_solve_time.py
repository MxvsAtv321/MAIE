"""
Test QP solve time performance.
"""
import pytest
import time
import numpy as np
import pandas as pd
from maie.portfolio.optimizer import qp_optimize
from maie.portfolio.exposures import sector_one_hot, beta_exposures

@pytest.mark.perf
def test_qp_solve_time():
    """Test that QP solve time is within acceptable bounds."""
    # Create synthetic data
    np.random.seed(42)
    n_assets = 50
    tickers = [f"SIM{i:03d}" for i in range(n_assets)]
    
    # Generate expected returns
    expected = pd.Series(np.random.randn(n_assets) * 0.01, index=tickers)
    
    # Generate previous weights
    prev_weights = pd.Series(np.random.randn(n_assets) * 0.02, index=tickers)
    prev_weights = prev_weights / prev_weights.abs().sum() * 0.5  # Normalize to 50% gross
    
    # Generate returns window for exposures
    returns_window = pd.DataFrame(
        np.random.randn(60, n_assets) * 0.02,
        index=pd.date_range("2024-01-01", periods=60),
        columns=tickers
    )
    
    # Generate exposures
    sector_exp = sector_one_hot(tickers)
    beta_exp = beta_exposures(returns_window)
    exposures = pd.concat([sector_exp, beta_exp.to_frame().T], axis=0)
    
    # Time the optimization
    start_time = time.perf_counter()
    
    weights = qp_optimize(
        expected=expected,
        prev_weights=prev_weights,
        returns_window=returns_window,
        constraints_yaml="constraints.yaml",
        exposures=exposures,
    )
    
    solve_time = time.perf_counter() - start_time
    
    # Assert solve time is reasonable (30ms threshold for synthetic)
    assert solve_time < 0.030, f"QP solve time {solve_time*1000:.1f}ms exceeds 30ms threshold"
    
    # Assert we got valid weights
    assert len(weights) == n_assets
    assert not weights.isna().any()
    assert abs(weights.sum()) < 0.01  # Roughly dollar neutral
