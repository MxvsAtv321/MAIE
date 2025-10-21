# TODO(cursor): load a persisted model (MLflow) instead of fitting on request; add /explain endpoint returning top features.
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from maie.features import build_features
from maie.models import StructuredModel


app = FastAPI(title="MAIE API", version="0.1.0")


class ScoreRequest(BaseModel):
    prices: Dict[str, List[float]]  # {ticker: [recent closes ...]}


class ScoreResponse(BaseModel):
    alpha: Dict[str, float]


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    # Build a tiny DataFrame from input
    lens = {k: len(v) for k, v in req.prices.items()}
    n = max(lens.values())
    idx = pd.RangeIndex(n)

    data = {k: pd.Series(v, index=pd.RangeIndex(len(v))) for k, v in req.prices.items()}
    df = pd.DataFrame(data).reindex(index=idx).ffill().bfill()

    X, y = build_features(df)
    # Fit a quick model on the provided history (MVP). See TODO in codebase for persistence.
    model = StructuredModel()
    model.fit(X, y)
    # Score latest row per asset: take last available date's features per (date, asset)
    last_idx = X.index.get_level_values(0).max()
    X_last = X.loc[last_idx]
    preds = model.predict(X_last)

    # Convert to dict
    alpha = {asset: float(pred) for asset, pred in preds.items()}
    return ScoreResponse(alpha=alpha)


