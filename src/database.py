"""SQLite prediction history. Privacy: full text/headline are stored ONLY when the caller opts in."""
from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

from . import config
from .data_processing import text_key

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    label TEXT NOT NULL CHECK (label IN ('FAKE','REAL')),
    score_real REAL NOT NULL,
    word_count INTEGER NOT NULL,
    text_hash TEXT NOT NULL,
    source TEXT,
    headline TEXT,
    text TEXT,
    text_stored INTEGER NOT NULL DEFAULT 0,
    model_version TEXT,
    latency_ms REAL,
    kind TEXT NOT NULL DEFAULT 'single'
);
CREATE INDEX IF NOT EXISTS idx_pred_created ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_pred_label ON predictions(label);
"""


class DatabaseError(RuntimeError):
    """Safe-to-display database failure."""


@contextmanager
def _connect(path=None):
    p = Path(path) if path else config.db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        with conn:  # one transaction: commit on success, rollback on error
            yield conn
    except sqlite3.Error as exc:
        raise DatabaseError("The history database could not be accessed.") from exc
    finally:
        conn.close()


def _row(label, score_real, text, headline, source, model_version, latency_ms, kind, store_text):
    clean = (text or "").strip()
    return (datetime.now(timezone.utc).replace(microsecond=0).isoformat(), label, float(score_real),
            len(clean.split()), hashlib.sha256(text_key(clean).encode("utf-8")).hexdigest(),
            (source or "").strip()[:200] or None,
            (headline or "").strip()[:500] if store_text and headline else None,
            clean if store_text else None, int(bool(store_text)), model_version,
            None if latency_ms is None else float(latency_ms), kind)


_INSERT = ("INSERT INTO predictions (created_at,label,score_real,word_count,text_hash,source,headline,text,"
           "text_stored,model_version,latency_ms,kind) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)")


def add_prediction(label, score_real, text, headline=None, source=None, model_version=None, latency_ms=None,
                   kind="single", store_text=False, db_path=None) -> int:
    with _connect(db_path) as conn:
        cur = conn.execute(_INSERT, _row(label, score_real, text, headline, source, model_version, latency_ms, kind, store_text))
        return int(cur.lastrowid)


def add_many(records: Iterable[dict], store_text=False, db_path=None) -> int:
    rows = [_row(r["label"], r["score_real"], r["text"], r.get("headline"), r.get("source"), r.get("model_version"),
                 None, "batch", store_text) for r in records]
    if not rows:
        return 0
    with _connect(db_path) as conn:
        conn.executemany(_INSERT, rows)
    return len(rows)


def _like(term: str) -> str:
    return "%" + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def query_history(date_from: Optional[str] = None, date_to: Optional[str] = None, labels: Optional[Sequence[str]] = None,
                  search: Optional[str] = None, limit: int = 500, offset: int = 0, db_path=None) -> list:
    """Dates are ISO strings (YYYY-MM-DD). Search covers source, headline and stored text."""
    where, args = [], []
    if date_from:
        where.append("created_at >= ?"); args.append(date_from)
    if date_to:
        where.append("created_at < date(?, '+1 day')"); args.append(date_to)
    if labels:
        where.append(f"label IN ({','.join('?' * len(labels))})"); args.extend(labels)
    if search and search.strip():
        t = _like(search.strip())
        where.append("(source LIKE ? ESCAPE '\\' OR headline LIKE ? ESCAPE '\\' OR text LIKE ? ESCAPE '\\')")
        args.extend([t, t, t])
    sql = "SELECT * FROM predictions" + (" WHERE " + " AND ".join(where) if where else "")
    sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
    with _connect(db_path) as conn:
        return [dict(r) for r in conn.execute(sql, [*args, int(max(1, min(limit, 5000))), int(max(0, offset))])]


def count(db_path=None) -> int:
    with _connect(db_path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])


def delete_ids(ids: Sequence[int], db_path=None) -> int:
    ids = [int(i) for i in ids]
    if not ids:
        return 0
    with _connect(db_path) as conn:
        return conn.execute(f"DELETE FROM predictions WHERE id IN ({','.join('?' * len(ids))})", ids).rowcount


def delete_all(db_path=None) -> int:
    with _connect(db_path) as conn:
        return conn.execute("DELETE FROM predictions").rowcount
