"""Metric computation. Pure functions - everything is computed from real predictions."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)


def class_balance(labels: Sequence[str]) -> dict:
    arr = np.asarray(list(labels))
    out = {str(k): int((arr == k).sum()) for k in sorted(set(arr.tolist()))}
    total = sum(out.values()) or 1
    return {"counts": out, "fractions": {k: round(v / total, 4) for k, v in out.items()}}


def compute_metrics(y_true: Sequence[str], y_pred: Sequence[str], score_positive: Optional[Sequence[float]],
                    classes: Sequence[str], positive_class: str = "REAL") -> dict:
    classes = list(classes)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    per_class = {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                 for i, c in enumerate(classes)}
    roc = None
    if score_positive is not None and len(set(y_true.tolist())) == 2:
        try:
            roc = float(roc_auc_score((y_true == positive_class).astype(int), np.asarray(score_positive)))
        except ValueError:
            roc = None
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": per_class,
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0)),
        "roc_auc": roc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "confusion_labels": classes,
        "classification_report": classification_report(y_true, y_pred, labels=classes, zero_division=0, digits=4),
    }
