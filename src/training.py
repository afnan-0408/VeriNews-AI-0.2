"""Training pipeline: split -> leakage check -> fit (TF-IDF + Logistic Regression) -> evaluate -> save.

TF-IDF lives inside a scikit-learn Pipeline and is fitted on the training split only.
Nothing is written to the artifact directory unless training and evaluation both succeed.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline

from . import config
from .data_processing import find_leakage
from .evaluation import class_balance, compute_metrics

ProgressFn = Callable[[float, str], None]


class TrainingError(RuntimeError):
    """Raised when training cannot proceed. Messages are safe to show to users."""


@dataclass
class TrainConfig:
    test_size: float = 0.2
    random_state: int = config.RANDOM_SEED
    max_features: int = 50_000
    ngram_range: tuple = (1, 2)
    min_df: int = 2
    C: float = 1.0
    max_iter: int = 1000
    class_weight: Optional[str] = "balanced"
    cv_folds: int = 5                      # 0 disables cross-validation
    max_rows: int = config.MAX_TRAIN_ROWS
    split_strategy: str = "random"         # random | group | time
    group_col: Optional[str] = None
    date_col: Optional[str] = None
    near_duplicate_threshold: float = 0.9
    remove_leaked_test_rows: bool = True

    def validate(self) -> None:
        if not 0.05 <= self.test_size <= 0.5:
            raise TrainingError("test_size must be between 0.05 and 0.5.")
        if self.split_strategy not in {"random", "group", "time"}:
            raise TrainingError("split_strategy must be 'random', 'group' or 'time'.")
        if self.split_strategy == "group" and not self.group_col:
            raise TrainingError("A group column is required for a grouped split.")
        if self.split_strategy == "time" and not self.date_col:
            raise TrainingError("A date column is required for a time-based split.")
        if self.C <= 0 or self.max_features < 10 or self.min_df < 1 or self.max_rows < 50:
            raise TrainingError("Invalid hyper-parameter value (C>0, max_features>=10, min_df>=1, max_rows>=50).")
        if not (1 <= self.ngram_range[0] <= self.ngram_range[1] <= 3):
            raise TrainingError("ngram_range must satisfy 1 <= low <= high <= 3.")


@dataclass
class TrainingResult:
    pipeline: Pipeline
    metadata: dict
    metrics: dict
    artifact_dir: Path


def build_pipeline(cfg: TrainConfig) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=cfg.max_features, ngram_range=tuple(cfg.ngram_range), min_df=cfg.min_df,
                                  sublinear_tf=True, strip_accents="unicode", lowercase=True)),
        ("clf", LogisticRegression(C=cfg.C, max_iter=cfg.max_iter, class_weight=cfg.class_weight,
                                   solver="liblinear", random_state=cfg.random_state)),
    ])


def _split(df: pd.DataFrame, cfg: TrainConfig):
    idx = np.arange(len(df))
    notes = {}
    if cfg.split_strategy == "random":
        tr, te = train_test_split(idx, test_size=cfg.test_size, random_state=cfg.random_state, stratify=df["label"])
        return np.sort(tr), np.sort(te), notes
    if cfg.split_strategy == "group":
        if cfg.group_col not in df.columns:
            raise TrainingError(f"Group column '{cfg.group_col}' is not in the dataset.")
        groups = df[cfg.group_col].fillna("__missing__").astype(str)
        if groups.nunique() < 3:
            raise TrainingError("Need at least 3 distinct groups for a grouped split.")
        gss = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state)
        tr, te = next(gss.split(idx, df["label"], groups=groups))
        return tr, te, notes
    if cfg.date_col not in df.columns:
        raise TrainingError(f"Date column '{cfg.date_col}' is not in the dataset.")
    dates = pd.to_datetime(df[cfg.date_col], errors="coerce", utc=True)
    valid = idx[dates.notna().to_numpy()]
    notes["rows_without_valid_date"] = int(len(idx) - len(valid))
    if len(valid) < 50:
        raise TrainingError("Too few rows with a parseable date for a time-based split.")
    order = valid[np.argsort(dates.to_numpy()[valid], kind="stable")]
    cut = int(len(order) * (1 - cfg.test_size))
    return order[:cut], order[cut:], notes


def _cross_validate(pipe: Pipeline, x: pd.Series, y: pd.Series, groups, cfg: TrainConfig) -> Optional[dict]:
    if cfg.cv_folds < 2:
        return None
    folds = min(cfg.cv_folds, int(y.value_counts().min()))
    if folds < 2:
        return None
    try:
        if groups is not None:
            cv = GroupKFold(n_splits=min(folds, groups.nunique()))
        else:
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg.random_state)
        res = cross_validate(clone(pipe), x, y, cv=cv, groups=groups, scoring=["accuracy", "f1_macro"], n_jobs=1)
    except ValueError as exc:
        return {"error": f"Cross-validation skipped: {exc}"}
    return {
        "folds": int(len(res["test_accuracy"])),
        "on": "training split only",
        "accuracy_mean": float(res["test_accuracy"].mean()), "accuracy_std": float(res["test_accuracy"].std()),
        "f1_macro_mean": float(res["test_f1_macro"].mean()), "f1_macro_std": float(res["test_f1_macro"].std()),
    }


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (Path, tuple, set)):
        return list(o) if not isinstance(o, Path) else str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def _atomic_write_text(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_artifacts(pipeline: Pipeline, metadata: dict, metrics: dict, artifact_dir: Path) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / config.MODEL_FILENAME
    fd, tmp = tempfile.mkstemp(dir=artifact_dir, suffix=".tmp")
    os.close(fd)
    try:
        joblib.dump(pipeline, tmp, compress=3)
        digest = sha256_file(Path(tmp))
        os.replace(tmp, model_path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    trained_at = datetime.fromisoformat(metadata["trained_at"])
    metadata["model_sha256"] = digest
    metadata["model_version"] = f"{trained_at:%Y%m%d-%H%M%S}-{digest[:8]}"
    metrics["model_version"] = metadata["model_version"]
    _atomic_write_text(artifact_dir / config.METRICS_FILENAME, json.dumps(metrics, indent=2, default=_json_default))
    # metadata is written last: its presence marks a complete artifact set
    _atomic_write_text(artifact_dir / config.METADATA_FILENAME, json.dumps(metadata, indent=2, default=_json_default))
    return metadata


LIMITATIONS = [
    "The model scores writing patterns learned from its training data; it does not verify facts.",
    "A random split can overestimate performance when related articles appear on both sides of the split.",
    "Datasets may carry source, topic, period and publisher bias that the model can exploit as shortcuts.",
    "Scores are uncalibrated model outputs, not probabilities of truth.",
]


def train_model(df: pd.DataFrame, cfg: Optional[TrainConfig] = None, artifact_dir=None,
                dataset_info: Optional[dict] = None, cleaning_report: Optional[dict] = None,
                is_demo: bool = False, progress: Optional[ProgressFn] = None) -> TrainingResult:
    cfg = cfg or TrainConfig()
    cfg.validate()
    artifact_dir = Path(artifact_dir) if artifact_dir else config.artifact_dir()
    say = progress or (lambda frac, msg: None)

    if not {"text", "label"} <= set(df.columns):
        raise TrainingError("Prepared dataset must contain 'text' and 'label' columns.")
    counts = df["label"].value_counts()
    if set(counts.index) != set(config.LABELS):
        raise TrainingError("Both FAKE and REAL examples are required.")
    if counts.min() < config.MIN_ROWS_PER_CLASS:
        raise TrainingError(f"Each class needs at least {config.MIN_ROWS_PER_CLASS} examples (smallest has {counts.min()}).")

    say(0.05, "Preparing data")
    subsampled_from = None
    if len(df) > cfg.max_rows:
        subsampled_from = len(df)
        df, _ = train_test_split(df, train_size=cfg.max_rows, stratify=df["label"], random_state=cfg.random_state)
    df = df.reset_index(drop=True)

    say(0.12, "Splitting into train and test sets")
    tr, te, split_notes = _split(df, cfg)
    train_df, test_df = df.iloc[tr], df.iloc[te]
    if set(train_df["label"]) != set(config.LABELS) or set(test_df["label"]) != set(config.LABELS):
        raise TrainingError("Train or test split is missing a class. This split strategy is not usable on this dataset "
                            "(for example, when the classes cover different date ranges). Try a random split.")

    say(0.22, "Checking for train/test leakage")
    exact, near = find_leakage(train_df["text"].tolist(), test_df["text"].tolist(), cfg.near_duplicate_threshold)
    leakage = {"test_rows_before": int(len(test_df)), "exact_duplicates_of_train": int(exact.sum()),
               "exact_or_near_duplicates_of_train": int(near.sum()),
               "near_duplicate_threshold": cfg.near_duplicate_threshold, "removed_from_test": 0}
    if cfg.remove_leaked_test_rows and near.any():
        test_df = test_df[~near]
        leakage["removed_from_test"] = int(near.sum())
        if len(test_df) < 10 or set(test_df["label"]) != set(config.LABELS):
            raise TrainingError("Too many test rows duplicate training rows; the dataset is too redundant to evaluate.")
    leakage["test_rows_after"] = int(len(test_df))

    x_train, y_train = train_df["text"], train_df["label"]
    x_test, y_test = test_df["text"], test_df["label"]
    pipe = build_pipeline(cfg)

    say(0.35, "Cross-validating on the training split")
    groups = train_df[cfg.group_col].fillna("__missing__").astype(str) if cfg.split_strategy == "group" else None
    cv = _cross_validate(pipe, x_train, y_train, groups, cfg)

    say(0.6, "Fitting TF-IDF + Logistic Regression on the training split")
    try:
        pipe.fit(x_train, y_train)
    except ValueError as exc:
        raise TrainingError(f"Model fitting failed: {exc}") from exc
    classes = list(pipe.named_steps["clf"].classes_)

    say(0.8, "Evaluating on the held-out test set")
    real_idx = classes.index("REAL")
    test_pred = pipe.predict(x_test)
    test_score = pipe.predict_proba(x_test)[:, real_idx]
    test_metrics = compute_metrics(y_test, test_pred, test_score, classes)
    train_metrics = compute_metrics(y_train, pipe.predict(x_train), None, classes)

    method = {"random": f"Stratified random hold-out split ({int((1 - cfg.test_size) * 100)}/{int(cfg.test_size * 100)})",
              "group": f"Grouped hold-out split by '{cfg.group_col}' (no group appears in both sets)",
              "time": f"Time-based hold-out: earliest {int((1 - cfg.test_size) * 100)}% train, latest "
                      f"{int(cfg.test_size * 100)}% test (by '{cfg.date_col}')"}[cfg.split_strategy]
    method += "; TF-IDF fitted on the training split only (scikit-learn Pipeline)."
    trained_at = datetime.now(timezone.utc).replace(microsecond=0)
    metrics = {
        "evaluation_method": method,
        "test": test_metrics,
        "train": {"accuracy": train_metrics["accuracy"], "macro_f1": train_metrics["macro_f1"],
                  "note": "Training-set scores are reported separately and are NOT a measure of generalisation."},
        "cross_validation": cv,
        "class_balance": {"train": class_balance(y_train), "test": class_balance(y_test)},
        "sizes": {"train": int(len(train_df)), "test": int(len(test_df))},
        "leakage": leakage,
        "is_demo": bool(is_demo),
        "calibrated": False,
    }
    metadata = {
        "schema_version": 1, "app": config.APP_NAME, "trained_at": trained_at.isoformat(),
        "algorithm": "TF-IDF + Logistic Regression (liblinear)", "classes": classes,
        "label_mapping": {c: i for i, c in enumerate(classes)}, "calibrated": False, "is_demo": bool(is_demo),
        "uses_title": bool((dataset_info or {}).get("uses_title", False)),
        "split": {"strategy": cfg.split_strategy, "test_size": cfg.test_size, "seed": cfg.random_state,
                  "n_train": int(len(train_df)), "n_test": int(len(test_df)), **split_notes},
        "dataset": {**(dataset_info or {}), "rows_used": int(len(df)), "subsampled_from": subsampled_from,
                    "cleaning": cleaning_report},
        "config": asdict(cfg),
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
        "limitations": LIMITATIONS,
    }
    say(0.92, "Saving artifacts")
    try:
        metadata = save_artifacts(pipe, metadata, metrics, artifact_dir)
    except OSError as exc:
        raise TrainingError(f"Could not save model artifacts: {exc}") from exc
    say(1.0, "Done")
    return TrainingResult(pipe, metadata, metrics, artifact_dir)
