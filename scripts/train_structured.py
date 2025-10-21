 #!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from maie.data.synthetic import generate_synthetic_prices
from maie.features.tabular import build_features


ART_DIR = Path("artifacts")
ART_DIR.mkdir(parents=True, exist_ok=True)
URI_FILE = ART_DIR / "structured_model_uri.txt"
FEATS_FILE = ART_DIR / "feature_names.json"


def main() -> None:
    prices = generate_synthetic_prices(tickers=[f"SIM{i:03d}" for i in range(200)], end="2024-12-31")
    X, y = build_features(prices)

    # Use only rows without NaNs
    Xn = X.dropna()
    y = y.loc[Xn.index]

    feat_names = list(Xn.columns)
    FEATS_FILE.write_text(json.dumps(feat_names))

    mlflow.set_tracking_uri("file:./mlruns")
    with mlflow.start_run(run_name="structured_baseline"):
        model = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
        )
        model.fit(Xn.values, y.values)

        mlflow.log_param("n_assets", prices.shape[1])
        mlflow.log_param("n_rows", int(Xn.shape[0]))
        mlflow.lightgbm.log_model(model, artifact_path="model")
        run_id = mlflow.active_run().info.run_id
        uri = f"runs:/{run_id}/model"
        URI_FILE.write_text(uri)
        print(f"Logged model to: {uri}")


if __name__ == "__main__":
    main()


