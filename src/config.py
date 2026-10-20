"""Central configuration. Every path/limit can be overridden with environment variables.

Paths are resolved lazily (functions) so tests and deployments can change
environment variables without re-importing the package.
"""
from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "VeriNews-AI"
APP_VERSION = "1.0.0"
ROOT_DIR = Path(__file__).resolve().parent.parent

LABELS = ("FAKE", "REAL")
RANDOM_SEED = 42

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
METRICS_FILENAME = "metrics.json"


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(int(os.getenv(name, default)), minimum)
    except (TypeError, ValueError):
        return default


def artifact_dir() -> Path:
    raw = os.getenv("VERINEWS_ARTIFACT_DIR")
    return Path(raw).expanduser().resolve() if raw else ROOT_DIR / "artifacts"


def data_dir() -> Path:
    raw = os.getenv("VERINEWS_DATA_DIR")
    return Path(raw).expanduser().resolve() if raw else ROOT_DIR / "data"


def db_path() -> Path:
    raw = os.getenv("VERINEWS_DB_PATH")
    return Path(raw).expanduser().resolve() if raw else data_dir() / "history.sqlite3"


def api_key() -> str:
    """Optional API key. Empty string means the API is open (fine for local use)."""
    return os.getenv("VERINEWS_API_KEY", "")


def pinned_model_sha256() -> str:
    """Optional: refuse to load any model file whose SHA-256 differs from this value."""
    return os.getenv("VERINEWS_MODEL_SHA256", "").strip().lower()


# Limits (all overridable)
MAX_UPLOAD_MB = _env_int("VERINEWS_MAX_UPLOAD_MB", 200)
MAX_TRAIN_ROWS = _env_int("VERINEWS_MAX_TRAIN_ROWS", 50_000, minimum=100)
MAX_BATCH_ROWS = _env_int("VERINEWS_MAX_BATCH_ROWS", 20_000)
MAX_API_BATCH = _env_int("VERINEWS_MAX_API_BATCH", 100)
MIN_WORDS = _env_int("VERINEWS_MIN_WORDS", 15)
MAX_TEXT_CHARS = _env_int("VERINEWS_MAX_TEXT_CHARS", 100_000, minimum=1000)
MIN_ROWS_PER_CLASS = 10
