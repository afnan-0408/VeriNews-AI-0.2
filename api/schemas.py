"""Pydantic request/response models for the REST API."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from src import config


class _Base(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), extra="forbid")


class PredictRequest(_Base):
    text: str = Field(..., min_length=1, max_length=config.MAX_TEXT_CHARS, description="News article text.")
    headline: Optional[str] = Field(None, max_length=500)


class BatchRequest(_Base):
    texts: List[str] = Field(..., min_length=1, max_length=config.MAX_API_BATCH)


class TermContribution(_Base):
    term: str
    contribution: float


class Explanation(_Base):
    toward_fake: List[TermContribution]
    toward_real: List[TermContribution]
    known_terms: int


class PredictResponse(_Base):
    label: Literal["FAKE", "REAL"]
    score_real: float
    score_fake: float
    calibrated: bool
    model_version: str
    is_demo: bool
    latency_ms: float
    explanation: Explanation
    disclaimer: str


class BatchItem(_Base):
    index: int
    label: Optional[Literal["FAKE", "REAL"]] = None
    score_real: Optional[float] = None
    score_fake: Optional[float] = None
    status: str


class BatchResponse(_Base):
    model_version: str
    is_demo: bool
    calibrated: bool = False
    results: List[BatchItem]


class ModelInfo(_Base):
    state: Literal["missing", "ready", "demo", "error"]
    message: str = ""
    model_version: Optional[str] = None
    trained_at: Optional[str] = None
    is_demo: Optional[bool] = None


class HealthResponse(_Base):
    status: str
    app_version: str
    model: ModelInfo
