"""Inference. The single place that loads and applies the saved pipeline (Streamlit and the API both use it)."""
from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional, Sequence

import joblib
import numpy as np

from . import config
from .training import sha256_file

DISCLAIMER = ("This is a machine-learning estimate of writing-style similarity to the training data. "
              "It is not verified proof that a story is true or false.")


class ModelNotAvailable(RuntimeError):
    """No usable trained model (missing, corrupt or failed integrity check)."""


class InvalidInput(ValueError):
    """The submitted text cannot be scored. Message is safe to show to users."""


class ModelBundle:
    def __init__(self, pipeline, metadata: dict, metrics: Optional[dict]):
        self.pipeline, self.metadata, self.metrics = pipeline, metadata, metrics
        self.classes = list(pipeline.named_steps["clf"].classes_)
        self.real_index = self.classes.index("REAL")

    @property
    def is_demo(self) -> bool:
        return bool(self.metadata.get("is_demo"))

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))


def _paths(artifact_dir):
    d = Path(artifact_dir) if artifact_dir else config.artifact_dir()
    return d, d / config.MODEL_FILENAME, d / config.METADATA_FILENAME, d / config.METRICS_FILENAME


@lru_cache(maxsize=4)
def _load_cached(dir_str: str, model_stamp: tuple, meta_stamp: int) -> ModelBundle:
    d, model_path, meta_path, metrics_path = _paths(dir_str)
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelNotAvailable("Model metadata is unreadable. Please retrain the model.") from exc
    digest = sha256_file(model_path)
    if digest != metadata.get("model_sha256"):
        raise ModelNotAvailable("Model file failed its integrity check (does not match metadata). Retrain the model.")
    pinned = config.pinned_model_sha256()
    if pinned and digest != pinned:
        raise ModelNotAvailable("Model file does not match the pinned VERINEWS_MODEL_SHA256 value; refusing to load it.")
    try:
        pipeline = joblib.load(model_path)
        bundle_ok = hasattr(pipeline, "predict_proba") and "tfidf" in pipeline.named_steps and "clf" in pipeline.named_steps
    except Exception as exc:  # noqa: BLE001 - any load failure is reported generically
        raise ModelNotAvailable("Model file could not be loaded. Retrain the model in this environment.") from exc
    if not bundle_ok:
        raise ModelNotAvailable("Model file has an unexpected structure. Retrain the model.")
    metrics = None
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    return ModelBundle(pipeline, metadata, metrics)


def load_bundle(artifact_dir=None) -> ModelBundle:
    """Load (and cache) the trained model. Only load artifacts you trained yourself."""
    d, model_path, meta_path, _ = _paths(artifact_dir)
    if not model_path.is_file() or not meta_path.is_file():
        raise ModelNotAvailable("No trained model found. Train one on the Dataset & Training page "
                                "(or run scripts/train_model.py).")
    st = model_path.stat()
    return _load_cached(str(d), (st.st_mtime_ns, st.st_size), meta_path.stat().st_mtime_ns)


def model_status(artifact_dir=None) -> dict:
    """Never raises. state: missing | ready | demo | error."""
    try:
        b = load_bundle(artifact_dir)
    except ModelNotAvailable as exc:
        missing = str(exc).startswith("No trained model")
        return {"state": "missing" if missing else "error", "message": str(exc), "metadata": None, "metrics": None}
    return {"state": "demo" if b.is_demo else "ready", "message": "", "metadata": b.metadata, "metrics": b.metrics}


def validate_text(text, min_words: Optional[int] = None) -> str:
    if not isinstance(text, str):
        raise InvalidInput("Text must be a string.")
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text).strip()
    if not cleaned:
        raise InvalidInput("Please enter some news text to analyse.")
    if len(cleaned) > config.MAX_TEXT_CHARS:
        raise InvalidInput(f"Text is too long (limit {config.MAX_TEXT_CHARS:,} characters).")
    need = config.MIN_WORDS if min_words is None else min_words
    if len(cleaned.split()) < need:
        raise InvalidInput(f"Please provide at least {need} words; very short text cannot be scored reliably.")
    return cleaned


def _model_input(bundle: ModelBundle, text: str, headline: Optional[str]) -> str:
    h = (headline or "").strip()
    return f"{h}. {text}" if (h and bundle.metadata.get("uses_title")) else text


def explain(bundle: ModelBundle, model_text: str, top_k: int = 8) -> dict:
    """Real explanation: per-term contribution = tf-idf weight x logistic-regression coefficient."""
    tfidf, clf = bundle.pipeline.named_steps["tfidf"], bundle.pipeline.named_steps["clf"]
    row = tfidf.transform([model_text]).tocoo()
    names = tfidf.get_feature_names_out()
    contrib = row.data * clf.coef_[0][row.col]            # positive -> pushes toward REAL
    order = np.argsort(contrib)
    toward_fake = [{"term": str(names[row.col[i]]), "contribution": float(contrib[i])} for i in order[:top_k] if contrib[i] < 0]
    toward_real = [{"term": str(names[row.col[i]]), "contribution": float(contrib[i])} for i in order[::-1][:top_k] if contrib[i] > 0]
    return {"toward_fake": toward_fake, "toward_real": toward_real, "known_terms": int(row.nnz)}


def _result(bundle, score_real, latency_ms, extra=None) -> dict:
    out = {"label": "REAL" if score_real >= 0.5 else "FAKE", "score_real": float(score_real),
           "score_fake": float(1 - score_real), "calibrated": False, "model_version": bundle.version,
           "is_demo": bundle.is_demo, "latency_ms": round(latency_ms, 2), "disclaimer": DISCLAIMER}
    out.update(extra or {})
    return out


def predict(bundle: ModelBundle, text: str, headline: Optional[str] = None, top_k: int = 8) -> dict:
    start = time.perf_counter()
    cleaned = validate_text(text)
    model_text = _model_input(bundle, cleaned, headline)
    exp = explain(bundle, model_text, top_k)
    if exp["known_terms"] == 0:
        raise InvalidInput("None of the words are in the model's vocabulary (is the text in another language?).")
    score_real = float(bundle.pipeline.predict_proba([model_text])[0][bundle.real_index])
    return _result(bundle, score_real, (time.perf_counter() - start) * 1000, {"explanation": exp})


def predict_batch(bundle: ModelBundle, texts: Sequence, chunk_size: int = 500,
                  progress: Optional[Callable[[float], None]] = None, min_words: Optional[int] = None) -> list:
    """Row-wise results; invalid rows are reported (status != 'ok') instead of aborting the batch."""
    results = [{"index": i, "label": None, "score_real": None, "score_fake": None, "status": "ok"} for i in range(len(texts))]
    valid_idx, valid_txt = [], []
    for i, t in enumerate(texts):
        try:
            valid_txt.append(validate_text("" if t is None or (not isinstance(t, str) and t != t) else str(t), min_words))
            valid_idx.append(i)
        except InvalidInput as exc:
            results[i]["status"] = f"skipped: {exc}"
    for s in range(0, len(valid_txt), chunk_size):
        part = valid_txt[s:s + chunk_size]
        scores = bundle.pipeline.predict_proba(part)[:, bundle.real_index]
        for j, sc in zip(valid_idx[s:s + chunk_size], scores):
            results[j].update(label="REAL" if sc >= 0.5 else "FAKE", score_real=float(sc), score_fake=float(1 - sc))
        if progress:
            progress(min(1.0, (s + chunk_size) / max(len(valid_txt), 1)))
    return results
