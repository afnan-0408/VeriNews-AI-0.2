"""Dataset loading, validation, label normalisation, cleaning and leakage detection.

Design rules: never silently treat unknown labels as REAL; keep article text
intact (only whitespace trimming); record every dropped row in a report.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config

FAKE, REAL = "FAKE", "REAL"
DEFAULT_LABEL_MAP = {
    "fake": FAKE, "false": FAKE, "hoax": FAKE, "fake news": FAKE, "misinformation": FAKE,
    "real": REAL, "true": REAL, "genuine": REAL, "legit": REAL, "legitimate": REAL, "real news": REAL,
}
# Numeric labels (0/1) are deliberately NOT in the default map: datasets disagree on which is which.

ALLOWED_SUFFIXES = {".csv", ".tsv", ".txt"}
_AGENCY_PREFIX = re.compile(r"^\s*[A-Z][A-Za-z .,'/&-]{0,40}\(\s*(?:Reuters|AP|AFP)\s*\)\s*[-\u2013\u2014]\s*")


class DatasetError(ValueError):
    """Raised for any problem with user-supplied data. Messages are safe to show to users."""


@dataclass
class CleaningReport:
    rows_before: int = 0
    rows_after: int = 0
    dropped_missing_text: int = 0
    dropped_too_short: int = 0
    dropped_unknown_label: int = 0
    dropped_label_conflicts: int = 0
    dropped_duplicates: int = 0
    unknown_label_values: list = field(default_factory=list)
    class_counts: dict = field(default_factory=dict)
    agency_prefix_stripped: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def text_key(value: str) -> str:
    """Normalised key used for exact-duplicate detection."""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def read_csv_safe(source, max_mb: Optional[int] = None, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Read an untrusted CSV/TSV (path, bytes or file-like). Parsing only - nothing is executed."""
    max_mb = config.MAX_UPLOAD_MB if max_mb is None else max_mb
    name = ""
    if isinstance(source, (str, Path)):
        path = Path(source)
        name = path.name
        if not path.is_file():
            raise DatasetError(f"File not found: {path.name}")
        if path.stat().st_size > max_mb * 1024 * 1024:
            raise DatasetError(f"File is larger than the {max_mb} MB limit.")
        raw = path.read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    else:
        name = getattr(source, "name", "") or ""
        raw = source.getvalue() if hasattr(source, "getvalue") else source.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
    if name and Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
        raise DatasetError("Unsupported file type. Please upload a .csv, .tsv or .txt file.")
    if len(raw) > max_mb * 1024 * 1024:
        raise DatasetError(f"File is larger than the {max_mb} MB limit.")
    if not raw.strip():
        raise DatasetError("The file is empty.")

    sample = raw[:4096].decode("utf-8", errors="ignore")
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    last_error: Optional[Exception] = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc, dtype=str, nrows=max_rows,
                               keep_default_na=True, on_bad_lines="skip")
        except UnicodeDecodeError as exc:
            last_error = exc
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise DatasetError("Could not parse the file as CSV. Check the delimiter and quoting.") from exc
    raise DatasetError("Could not decode the file. Please save it as UTF-8.") from last_error


def normalize_labels(series: Iterable, label_map: Optional[Mapping[str, str]] = None):
    """Map raw labels to FAKE/REAL. Returns (list with None for unknown, sorted unknown raw values)."""
    mapping = {str(k).strip().lower(): str(v).strip().upper() for k, v in (label_map or DEFAULT_LABEL_MAP).items()}
    bad = set(mapping.values()) - {FAKE, REAL}
    if bad:
        raise DatasetError(f"Label map may only target FAKE or REAL (got {sorted(bad)}).")
    out, unknown = [], set()
    for v in series:
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            out.append(None)
            continue
        key = str(v).strip().lower()
        if key in mapping:
            out.append(mapping[key])
        else:
            out.append(None)
            unknown.add(str(v).strip())
    return out, sorted(unknown)


def load_isot_pair(fake_source, true_source, max_rows_each: Optional[int] = None) -> pd.DataFrame:
    """Combine ISOT-style Fake.csv and True.csv (columns title, text, subject, date)."""
    frames = []
    for src, label in ((fake_source, FAKE), (true_source, REAL)):
        df = read_csv_safe(src, max_rows=max_rows_each)
        if "text" not in df.columns:
            raise DatasetError(f"The {label} file must contain a 'text' column (found: {list(df.columns)[:8]}).")
        df = df.copy()
        df["label"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def prepare_dataset(df: pd.DataFrame, text_col: str, label_col: str, title_col: Optional[str] = None,
                    label_map: Optional[Mapping[str, str]] = None, min_chars: int = 20,
                    strip_agency_prefix: bool = False, keep_cols: Sequence[str] = ()):
    """Validate + clean. Returns (clean_df[text,label,+keep_cols], CleaningReport)."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise DatasetError("The dataset is empty.")
    needed = [text_col, label_col] + ([title_col] if title_col else []) + list(keep_cols)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise DatasetError(f"Column(s) not found in dataset: {missing}")
    if text_col == label_col:
        raise DatasetError("The text column and label column must be different.")

    report = CleaningReport(rows_before=len(df), agency_prefix_stripped=strip_agency_prefix)

    def clean_cell(v) -> str:
        return "" if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v).strip()

    body = [clean_cell(v) for v in df[text_col]]
    if strip_agency_prefix:
        body = [_AGENCY_PREFIX.sub("", b, count=1) for b in body]
    if title_col:
        titles = [clean_cell(v) for v in df[title_col]]
        model_text = [f"{t}. {b}" if (t and b) else b for t, b in zip(titles, body)]
    else:
        model_text = body

    labels, unknown = normalize_labels(df[label_col].tolist(), label_map)
    frame = pd.DataFrame({"text": model_text, "label": labels, "_body_len": [len(b) for b in body]})
    for c in keep_cols:
        frame[c] = df[c].to_numpy()

    keep = frame["_body_len"] > 0
    report.dropped_missing_text = int((~keep).sum())
    frame = frame[keep]

    keep = frame["_body_len"] >= min_chars
    report.dropped_too_short = int((~keep).sum())
    frame = frame[keep]

    keep = frame["label"].notna()
    report.dropped_unknown_label = int((~keep).sum())
    report.unknown_label_values = unknown[:50]
    frame = frame[keep].copy()

    frame["_key"] = [text_key(t) for t in frame["text"]]
    n_labels = frame.groupby("_key")["label"].transform("nunique")
    conflict = n_labels > 1
    report.dropped_label_conflicts = int(conflict.sum())
    frame = frame[~conflict]

    dup = frame.duplicated(subset="_key", keep="first")
    report.dropped_duplicates = int(dup.sum())
    frame = frame[~dup].drop(columns=["_key", "_body_len"]).reset_index(drop=True)

    report.rows_after = len(frame)
    report.class_counts = {k: int(v) for k, v in frame["label"].value_counts().items()}
    if frame.empty:
        raise DatasetError("No usable rows remain after cleaning. Check the column mapping and label values.")
    if frame["label"].nunique() < 2:
        raise DatasetError("Only one class (FAKE or REAL) remains after cleaning; both are required.")
    return frame, report


def find_leakage(train_texts: Sequence[str], test_texts: Sequence[str], near_threshold: float = 0.9, chunk: int = 100):
    """Boolean arrays over test_texts: (exact duplicate of a train text, exact-or-near duplicate).

    Near-duplicates use TF-IDF cosine similarity (vectoriser fitted on the training texts only).
    """
    train_keys = {text_key(t) for t in train_texts}
    exact = np.array([text_key(t) in train_keys for t in test_texts], dtype=bool)
    near = exact.copy()
    if len(train_texts) and len(test_texts):
        try:
            vec = TfidfVectorizer(max_features=20000, sublinear_tf=True)
            x_train = vec.fit_transform(train_texts)
            x_test = vec.transform(test_texts)
            for start in range(0, len(test_texts), chunk):
                sims = x_test[start:start + chunk] @ x_train.T
                best = np.asarray(sims.max(axis=1).todense()).ravel()
                near[start:start + chunk] |= best >= near_threshold
        except ValueError:  # empty vocabulary etc. - diagnostics only
            pass
    return exact, near
