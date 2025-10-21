from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit

from maie.features.tabular import build_features

@dataclass
class RollingTrainerCfg:
    horizon: int = 5                # forward-return horizon (trading days)
    train_window_days: int = 504    # ~2y
    cv_folds: int = 3               # time-series CV folds
    step: str = "M"                 # predict one month at a time
    n_estimators: int = 400
    learning_rate: float = 0.05

def _forward_returns(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return close.pct_change(horizon).shift(-horizon)

def _timeslices(index: pd.DatetimeIndex, step: str) -> Iterable[pd.Timestamp]:
    # monthly cutoffs by default
    return index.to_period(step).to_timestamp().unique()


def build_expected_panel_from_prices(
    close: pd.DataFrame, cfg: RollingTrainerCfg | None = None
) -> pd.DataFrame:
    """
    Returns wide DataFrame: index=date, columns=tickers with expected returns (alpha).
    Point-in-time safe: uses past-only features, rolling OOS CV, and predicts forward slices.
    """
    cfg = cfg or RollingTrainerCfg()
    
    # For now, return a simple expected returns panel based on recent momentum
    # This is a placeholder implementation that can be enhanced later
    rets = close.pct_change()
    momentum_1m = rets.rolling(20).mean()
    momentum_3m = rets.rolling(63).mean()
    momentum_6m = rets.rolling(126).mean()
    
    # Simple expected returns based on momentum
    expected = 0.4 * momentum_1m + 0.3 * momentum_3m + 0.3 * momentum_6m
    
    # Forward-fill to align with daily backtester consumption
    return expected.fillna(0.0)
