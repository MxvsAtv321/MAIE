# TODO(cursor): load a persisted model (MLflow) instead of fitting on request; add /explain endpoint returning top features.
from __future__ import annotations

from typing import Dict, List
import os
import json

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from maie.features import build_features
from maie.models import StructuredModel
import mlflow
import mlflow.pyfunc


app = FastAPI(title="MAIE API", version="0.1.0")


class ScoreRequest(BaseModel):
    prices: Dict[str, List[float]]  # {ticker: [recent closes ...]}


class ScoreResponse(BaseModel):
    alpha: Dict[str, float]


class ExplainResponse(BaseModel):
    feature_importance: Dict[str, float]


# Load persisted model on startup (MLflow)
MODEL_URI = os.environ.get("MLFLOW_MODEL_URI", "").strip()
if not MODEL_URI and os.path.exists("artifacts/structured_model_uri.txt"):
    MODEL_URI = open("artifacts/structured_model_uri.txt").read().strip()

FEATURES: List[str] = []
if os.path.exists("artifacts/feature_names.json"):
    try:
        FEATURES = json.loads(open("artifacts/feature_names.json").read())
    except Exception:
        FEATURES = []

ML_MODEL = None
try:
    if MODEL_URI:
        ML_MODEL = mlflow.pyfunc.load_model(MODEL_URI)
except Exception:
    ML_MODEL = None


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    # Build a tiny DataFrame from input
    lens = {k: len(v) for k, v in req.prices.items()}
    n = max(lens.values())
    idx = pd.RangeIndex(n)

    data = {k: pd.Series(v, index=pd.RangeIndex(len(v))) for k, v in req.prices.items()}
    df = pd.DataFrame(data).reindex(index=idx).ffill().bfill()

    X, y = build_features(df)
    # Score latest row per asset: use persisted model if available
    last_idx = X.index.get_level_values(0).max()
    X_last = X.loc[last_idx]
    if ML_MODEL is not None and FEATURES:
        X_aligned = X_last.reindex(columns=FEATURES).fillna(0.0)
        pred = ML_MODEL.predict(X_aligned.values)
        alpha = {asset: float(v) for asset, v in zip(X_aligned.index.tolist(), pred)}
    else:
        model = StructuredModel()
        model.fit(X, y)
        preds = model.predict(X_last)
        alpha = {asset: float(pred) for asset, pred in preds.items()}
    return ScoreResponse(alpha=alpha)


@app.get("/explain", response_model=ExplainResponse)
def explain() -> ExplainResponse:
    out: Dict[str, float] = {}
    try:
        if ML_MODEL and hasattr(ML_MODEL._model_impl, "get_booster"):
            booster = ML_MODEL._model_impl.get_booster()
            importance = booster.feature_importance(importance_type="gain")
            names = booster.feature_name()
            out = {n: float(v) for n, v in zip(names, importance)}
    except Exception:
        pass
    return ExplainResponse(feature_importance=out)


