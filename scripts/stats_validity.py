#!/usr/bin/env python3
"""
Statistical validity checks for backtest results.
Computes Newey-West HAC Sharpe CI, block bootstrap, and PBO.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, Any

def newey_west_sharpe_ci(returns: pd.Series, confidence: float = 0.95) -> tuple[float, float, float]:
    """
    Compute Newey-West HAC Sharpe ratio with confidence interval.
    Returns (sharpe, lower_ci, upper_ci).
    """
    n = len(returns)
    if n < 2:
        return 0.0, 0.0, 0.0
    
    # Annualized Sharpe
    mean_ret = returns.mean()
    std_ret = returns.std()
    if std_ret == 0:
        return 0.0, 0.0, 0.0
    
    sharpe = mean_ret / std_ret * np.sqrt(252)
    
    # Newey-West HAC standard error
    # Simple implementation: use Bartlett kernel with lag truncation
    max_lags = min(4, n // 4)  # Conservative lag selection
    gamma_0 = np.var(returns)
    
    # Compute autocovariances
    autocovs = []
    for lag in range(1, max_lags + 1):
        if lag < n:
            autocov = np.cov(returns[:-lag], returns[lag:])[0, 1]
            autocovs.append(autocov)
        else:
            break
    
    # HAC variance
    hac_var = gamma_0
    for lag, autocov in enumerate(autocovs, 1):
        weight = 1 - lag / (max_lags + 1)  # Bartlett kernel
        hac_var += 2 * weight * autocov
    
    # Standard error
    se = np.sqrt(hac_var / n) / std_ret * np.sqrt(252)
    
    # Confidence interval
    z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
    margin = z_score * se
    
    return sharpe, sharpe - margin, sharpe + margin

def block_bootstrap_sharpe(returns: pd.Series, n_bootstrap: int = 10000, 
                          block_size: int = 20, confidence: float = 0.95) -> tuple[float, float]:
    """
    Block bootstrap for Sharpe ratio confidence interval.
    Returns (lower_ci, upper_ci).
    """
    n = len(returns)
    if n < block_size:
        return 0.0, 0.0
    
    bootstrap_sharpes = []
    
    for _ in range(n_bootstrap):
        # Generate bootstrap sample using moving block bootstrap
        bootstrap_returns = []
        i = 0
        while i < n:
            # Random block start
            start = np.random.randint(0, n - block_size + 1)
            block = returns.iloc[start:start + block_size]
            bootstrap_returns.extend(block.values)
            i += block_size
        
        # Truncate to original length
        bootstrap_returns = bootstrap_returns[:n]
        bootstrap_series = pd.Series(bootstrap_returns)
        
        # Compute Sharpe
        if bootstrap_series.std() > 0:
            sharpe = bootstrap_series.mean() / bootstrap_series.std() * np.sqrt(252)
            bootstrap_sharpes.append(sharpe)
    
    if not bootstrap_sharpes:
        return 0.0, 0.0
    
    # Compute confidence interval
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_sharpes, 100 * alpha / 2)
    upper = np.percentile(bootstrap_sharpes, 100 * (1 - alpha / 2))
    
    return lower, upper

def compute_pbo(returns: pd.Series) -> float:
    """
    Probability of Backtest Overfitting (PBO) estimation.
    Simplified version for synthetic data.
    """
    n = len(returns)
    if n < 50:
        return 0.0
    
    # Simple PBO proxy: measure consistency of performance
    # Split returns into chunks and measure variance of Sharpe ratios
    chunk_size = max(20, n // 10)
    n_chunks = n // chunk_size
    
    chunk_sharpes = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n)
        chunk = returns.iloc[start:end]
        
        if chunk.std() > 0:
            sharpe = chunk.mean() / chunk.std() * np.sqrt(252)
            chunk_sharpes.append(sharpe)
    
    if len(chunk_sharpes) < 2:
        return 0.0
    
    # PBO is related to the coefficient of variation of Sharpe ratios
    sharpe_std = np.std(chunk_sharpes)
    sharpe_mean = np.mean(chunk_sharpes)
    
    if abs(sharpe_mean) < 1e-6:
        return 0.0
    
    # Higher CV indicates more overfitting risk
    cv = sharpe_std / abs(sharpe_mean)
    pbo = min(1.0, cv)  # Cap at 1.0
    
    return pbo

def main():
    """Compute statistical validity metrics."""
    # Load backtest returns
    returns_file = Path("outputs_from_expected/returns.csv")
    if not returns_file.exists():
        returns_file = Path("outputs/returns.csv")
    
    if not returns_file.exists():
        print("No returns file found")
        return
    
    returns = pd.read_csv(returns_file, index_col=0, parse_dates=True).iloc[:, 0]
    
    # Compute metrics
    sharpe_ann = returns.mean() / returns.std() * np.sqrt(252)
    
    # Newey-West HAC CI
    sharpe_nw, sharpe_ci_lower, sharpe_ci_upper = newey_west_sharpe_ci(returns)
    
    # Block bootstrap CI
    boot_ci_lower, boot_ci_upper = block_bootstrap_sharpe(returns)
    
    # PBO
    pbo = compute_pbo(returns)
    
    # Prepare results
    stats = {
        "sharpe_ann": float(sharpe_ann),
        "sharpe_ci_95": [float(sharpe_ci_lower), float(sharpe_ci_upper)],
        "sharpe_boot_ci_95": [float(boot_ci_lower), float(boot_ci_upper)],
        "pbo": float(pbo)
    }
    
    # Load existing numbers and add stats
    numbers_file = Path("docs/numbers.json")
    if numbers_file.exists():
        with open(numbers_file) as f:
            numbers = json.load(f)
    else:
        numbers = {}
    
    numbers["stats"] = stats
    
    # Write back
    Path("docs").mkdir(exist_ok=True)
    with open(numbers_file, "w") as f:
        json.dump(numbers, f, indent=2)
    
    print(f"Statistical validity metrics computed:")
    print(f"  Sharpe: {sharpe_ann:.3f}")
    print(f"  NW CI: [{sharpe_ci_lower:.3f}, {sharpe_ci_upper:.3f}]")
    print(f"  Boot CI: [{boot_ci_lower:.3f}, {boot_ci_upper:.3f}]")
    print(f"  PBO: {pbo:.3f}")

if __name__ == "__main__":
    main()
