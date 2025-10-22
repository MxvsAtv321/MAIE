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
from maie.models.expected_io import load_expected_latest
import mlflow
import mlflow.pyfunc
import mlflow.lightgbm
import shap
from functools import lru_cache
from prometheus_fastapi_instrumentator import Instrumentator
from maie.config import settings


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

class ScoreExpectedRequest(BaseModel):
    tickers: List[str] | None = None  # if None, return all available


def _resolve_model_uri() -> str | None:
    # 1) env
    if settings.MLFLOW_MODEL_URI:
        return settings.MLFLOW_MODEL_URI.strip()
    # 2) artifacts file relative to repo root / this file
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "..", "artifacts", "structured_model_uri.txt"),
        os.path.join("artifacts", "structured_model_uri.txt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return open(p).read().strip()
    return None

MODEL_URI = _resolve_model_uri()

def _resolve_feature_names() -> list[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "..", "artifacts", "feature_names.json"),
        os.path.join("artifacts", "feature_names.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return json.loads(open(p).read())
    return []

FEATURES = _resolve_feature_names()

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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/ready")
def ready() -> dict[str, str]:
    """Readiness probe: model optional (configurable), expected_latest presence helpful."""
    model_ok = (ML_MODEL is not None) or (not settings.READINESS_REQUIRE_MODEL)
    expected_ok = True
    try:
        _ = load_expected_latest(settings.EXPECTED_DIR)
    except Exception:
        expected_ok = False
    status = "ready" if (model_ok or expected_ok) else "not_ready"
    return {"status": status, "model_loaded": str(bool(ML_MODEL)), "expected_available": str(expected_ok)}

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
    
    # Align to training feature order (from feature_names.json)
    if FEATURES:
        latest = latest.reindex(columns=FEATURES).fillna(0.0)
    else:
        latest = latest.fillna(0.0)
    
    xrow = latest.loc[[req.ticker]]
    explainer = _explainer_cached()
    if explainer is None:
        # Graceful fallback: return top absolute standardized features (no SHAP)
        s = xrow.iloc[0].abs().sort_values(ascending=False)
        pairs = list(zip(s.index[:req.top_k].tolist(), s.values[:req.top_k].astype(float)))
        return ExplainLocalResponse(ticker=req.ticker, top_features=pairs)
    
    xrow_values = xrow.values.reshape(1, -1)
    
    sv = explainer.shap_values(xrow_values)
    # LightGBM single output → sv is (n, d) or list; handle both
    if isinstance(sv, list):
        sv = sv[0]
    vals = sv[0]
    feats = latest.columns.tolist()
    pairs = sorted(zip(feats, map(float, vals)), key=lambda z: abs(z[1]), reverse=True)[:req.top_k]
    return ExplainLocalResponse(ticker=req.ticker, top_features=pairs)

@app.post("/score_expected", response_model=ScoreResponse)
def score_expected(req: ScoreExpectedRequest) -> ScoreResponse:
    """Return latest expected alphas from `expected/expected_latest.parquet`."""
    try:
        latest = load_expected_latest(settings.EXPECTED_DIR)  # 1-row snapshot
    except Exception:
        return ScoreResponse(alpha={})
    row = latest.iloc[-1]
    if req.tickers:
        row = row.reindex(req.tickers).dropna()
    return ScoreResponse(alpha=row.astype(float).to_dict())

# Prometheus metrics (enabled by default)
if settings.METRICS_ENABLED:
    Instrumentator().instrument(app).expose(app, include_in_schema=False)
