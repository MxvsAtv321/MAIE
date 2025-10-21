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
import mlflow.lightgbm
import shap
from functools import lru_cache


app = FastAPI(title="MAIE API", version="0.1.0")


class ScoreRequest(BaseModel):
    prices: Dict[str, List[float]]  # {ticker: [recent closes ...]}


class ScoreResponse(BaseModel):
    alpha: Dict[str, float]


class ExplainResponse(BaseModel):
    feature_importance: Dict[str, float]

class ExplainLocalRequest(BaseModel):
    prices: Dict[str, List[float]]
    ticker: str
    top_k: int = 10

class ExplainLocalResponse(BaseModel):
    ticker: str
    top_features: List[tuple[str, float]]  # (feature, shap_value)


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

ML_MODEL = None  # pyfunc
LGBM_MODEL = None  # native LightGBM model
try:
    if MODEL_URI:
        # Load both pyfunc (for prediction API) and native LightGBM flavor (for importance)
        ML_MODEL = mlflow.pyfunc.load_model(MODEL_URI)
        try:
            LGBM_MODEL = mlflow.lightgbm.load_model(MODEL_URI)
        except Exception:
            LGBM_MODEL = None
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
        # Prefer native LightGBM model if available
        if LGBM_MODEL is not None:
            pred = LGBM_MODEL.predict(X_aligned.values)
        else:
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
        # Prefer native LightGBM estimator to access importances reliably
        if LGBM_MODEL is not None:
            if hasattr(LGBM_MODEL, "feature_importances_"):
                names = FEATURES or [f"f{i}" for i in range(len(LGBM_MODEL.feature_importances_))]
                out = {n: float(v) for n, v in zip(names, LGBM_MODEL.feature_importances_)}
            elif hasattr(LGBM_MODEL, "booster_"):
                booster = LGBM_MODEL.booster_
                importance = booster.feature_importance(importance_type="gain")
                names = booster.feature_name()
                out = {n: float(v) for n, v in zip(names, importance)}
        elif ML_MODEL and hasattr(ML_MODEL, "_model_impl") and hasattr(ML_MODEL._model_impl, "get_booster"):
            booster = ML_MODEL._model_impl.get_booster()
            importance = booster.feature_importance(importance_type="gain")
            names = booster.feature_name()
            out = {n: float(v) for n, v in zip(names, importance)}
    except Exception:
        pass
    return ExplainResponse(feature_importance=out)

@lru_cache(maxsize=1)
def _explainer_cached():
    """Cache a TreeExplainer for speed."""
    booster = None
    if LGBM_MODEL is not None and hasattr(LGBM_MODEL, "booster_"):
        booster = LGBM_MODEL.booster_
    elif ML_MODEL and hasattr(ML_MODEL, "_model_impl") and hasattr(ML_MODEL._model_impl, "get_booster"):
        booster = ML_MODEL._model_impl.get_booster()
    if booster is None:
        return None
    return shap.TreeExplainer(booster)

@app.post("/explain_local", response_model=ExplainLocalResponse)
def explain_local(req: ExplainLocalRequest) -> ExplainLocalResponse:
    """Return top-K SHAP feature contributions for a given ticker on the latest row."""
    # Build DataFrame from input prices
    lens = {k: len(v) for k, v in req.prices.items()}
    n = max(lens.values())
    idx = pd.RangeIndex(n)
    
    data = {k: pd.Series(v, index=pd.RangeIndex(len(v))) for k, v in req.prices.items()}
    df = pd.DataFrame(data).reindex(index=idx).ffill().bfill()
    
    X, y = build_features(df)
    latest = X.iloc[-1]  # Get latest row
    
    if req.ticker not in latest.index:
        return ExplainLocalResponse(ticker=req.ticker, top_features=[])
    
    xrow = latest.loc[[req.ticker]].values.reshape(1, -1)
    explainer = _explainer_cached()
    if explainer is None:
        return ExplainLocalResponse(ticker=req.ticker, top_features=[])
    
    sv = explainer.shap_values(xrow)
    # LightGBM single output → sv is (n, d) or list; handle both
    if isinstance(sv, list):
        sv = sv[0]
    vals = sv[0]
    feats = latest.columns.tolist()
    pairs = sorted(zip(feats, map(float, vals)), key=lambda z: abs(z[1]), reverse=True)[:req.top_k]
    return ExplainLocalResponse(ticker=req.ticker, top_features=pairs)


