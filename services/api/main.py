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
import time
from prometheus_client import Counter, Gauge, Histogram


app = FastAPI(title="MAIE API", version="0.1.0")

# Custom metrics
EXPLAIN_FALLBACK = Counter("maie_explain_local_fallback_total", "Explain local path usage", ["kind"])
QP_INFEASIBLE = Counter("maie_qp_infeasible_total", "Count of QP infeasible days observed")
EXPECTED_TS = Gauge("maie_expected_latest_timestamp", "Unix timestamp of expected_latest.parquet")
QP_INFEASIBLE_RATIO = Gauge("maie_qp_infeasible_ratio", "Ratio of days with QP infeasible solutions")
QP_SOLVE = Histogram("maie_qp_solve_seconds", "QP solve duration seconds")
FEAT_SKEW = Counter("maie_feature_skew_total", "Count of feature alignment fixes")
maie_api_request_duration = Histogram('maie_api_request_duration_seconds', 'API request duration', ['endpoint'])

# Initialize QP infeasibility ratio from backtest metrics
try:
    import json, pathlib
    p = pathlib.Path("outputs_from_expected/metrics.json")
    if p.exists():
        meta = json.loads(p.read_text())
        nd = max(1, int(meta.get("n_days", 0)))
        QP_INFEASIBLE_RATIO.set(float(meta.get("infeasible_days", 0)) / nd)
except Exception:
    pass


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

def _get_booster():
    """Return a native LightGBM Booster if available, else None."""
    if LGBM_MODEL is not None and hasattr(LGBM_MODEL, "booster_"):
        return LGBM_MODEL.booster_
    if ML_MODEL is not None and hasattr(ML_MODEL._model_impl, "get_booster"):
        try:
            return ML_MODEL._model_impl.get_booster()
        except Exception:
            return None
    return None

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
    """Top-K per-ticker feature contributions on latest row (robust to lookback/NaNs)."""
    # 0) Canonicalize ticker casing (avoid "sim0001" vs "SIM0001" mismatches)
    want = str(req.ticker).strip()
    want_upper = want.upper()

    # 1) Build features from the raw price history payload
    df = pd.DataFrame({k: pd.Series(v) for k, v in req.prices.items()})
    # Standardize incoming keys to upper for alignment
    df.columns = [str(c).upper() for c in df.columns]

    X, y = build_features(df)                    # Returns (features, targets)
    # X has MultiIndex (date, ticker) with features as columns
    # Get the latest date slice
    latest_date = X.index.get_level_values(0).max()
    latest = X.loc[latest_date]                  # DataFrame with tickers as index, features as columns
    X = latest.copy()                            # rows=tickers, cols=features
    X.index = [str(ix).upper() for ix in X.index]       # upper-case index for matching

    # 2) Align columns to training feature order if we have it
    if FEATURES:
        if list(X.columns) != FEATURES:
            FEAT_SKEW.inc()
        X = X.reindex(columns=FEATURES)

    # 3) GUARANTEE the requested ticker has a row:
    #    if it was dropped by the feature builder (e.g., not enough lookback),
    #    create a zero row so we can still produce a stable explanation.
    if want_upper not in X.index:
        # Add a zeroed-out row for the missing ticker
        X.loc[want_upper] = 0.0

    # Fill any remaining gaps (NaNs → 0)
    X = X.fillna(0.0)

    # Keep only the requested ticker
    xrow = X.loc[[want_upper]]
    features_order = xrow.columns.tolist()

        # 4) Fast path: LightGBM native SHAP via pred_contrib
        booster = _get_booster()
        if booster is not None:
            try:
                contrib = booster.predict(xrow.values, pred_contrib=True)  # shape: (1, d+1) w/ bias last
                vals = contrib[0][:-1]  # drop bias term
                pairs = sorted(
                    zip(features_order, map(float, vals)),
                    key=lambda z: abs(z[1]),
                    reverse=True
                )[:req.top_k]
                EXPLAIN_FALLBACK.labels(kind="pred_contrib").inc()
                return ExplainLocalResponse(ticker=want, top_features=pairs)
            except Exception:
                pass  # fall through to slower paths

        # 5) Slow path: SHAP TreeExplainer if available
        explainer = _explainer_cached()
        if explainer is not None:
            try:
                sv = explainer.shap_values(xrow.values)
                if isinstance(sv, list):
                    sv = sv[0]
                vals = sv[0]
                pairs = sorted(
                    zip(features_order, map(float, vals)),
                    key=lambda z: abs(z[1]),
                    reverse=True
                )[:req.top_k]
                EXPLAIN_FALLBACK.labels(kind="tree").inc()
                return ExplainLocalResponse(ticker=want, top_features=pairs)
            except Exception:
                pass

    # 6) Guaranteed fallback: magnitude ranking of standardized features
    mags = xrow.iloc[0].abs().sort_values(ascending=False)
    pairs = list(zip(mags.index[:req.top_k].tolist(), mags.values[:req.top_k].astype(float)))
    EXPLAIN_FALLBACK.labels(kind="magnitude").inc()
    return ExplainLocalResponse(ticker=want, top_features=pairs)

@app.post("/score_expected", response_model=ScoreResponse)
def score_expected(req: ScoreExpectedRequest) -> ScoreResponse:
    """Return latest expected alphas from `expected/expected_latest.parquet`."""
    start_time = time.time()
    try:
        latest = load_expected_latest(settings.EXPECTED_DIR)  # 1-row snapshot
        # update freshness gauge
        EXPECTED_TS.set(float(latest.index.max().timestamp()))
    except Exception:
        return ScoreResponse(alpha={})
    row = latest.iloc[-1]
    if req.tickers:
        row = row.reindex(req.tickers).dropna()
    
    return ScoreResponse(alpha=row.astype(float).to_dict())
    finally:
        duration = time.time() - start_time
        maie_api_request_duration.labels(endpoint="score_expected").observe(duration)

# Prometheus metrics (enabled by default)
if settings.METRICS_ENABLED:
    Instrumentator().instrument(app).expose(app, include_in_schema=False)
