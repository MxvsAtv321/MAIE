# TODO(cursor): replace toy expanding split with robust rolling-window CV and OOS prediction pipeline; persist model state per date.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class TrainResult:
    model: LGBMRegressor
    oos_r2: float


class StructuredModel:
    """
    Simple LightGBM regressor with time-series split.

    TODO(cursor): replace toy expanding split with robust rolling-window CV and OOS prediction pipeline; persist model state per date.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.model: Optional[LGBMRegressor] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TrainResult:
        X_np = X.values
        y_np = y.values

        tscv = TimeSeriesSplit(n_splits=3)
        best_score = -np.inf
        best_model: Optional[LGBMRegressor] = None

        for train_idx, val_idx in tscv.split(X_np):
            X_tr, X_va = X_np[train_idx], X_np[val_idx]
            y_tr, y_va = y_np[train_idx], y_np[val_idx]

            model = LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_va)
            score = r2_score(y_va, preds)
            if score > best_score:
                best_score = score
                best_model = model

        assert best_model is not None
        self.model = best_model
        return TrainResult(model=best_model, oos_r2=float(best_score))

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not fit")
        preds = self.model.predict(X.values)
        return pd.Series(preds, index=X.index)


