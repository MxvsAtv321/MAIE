from __future__ import annotations
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    LOG_LEVEL: str = "INFO"
    MLFLOW_MODEL_URI: str | None = None
    EXPECTED_DIR: str = "expected"
    METRICS_ENABLED: bool = True
    READINESS_REQUIRE_MODEL: bool = False  # allow API to be ready without model if using expected panel only

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


settings = Settings()
