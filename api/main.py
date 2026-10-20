"""Optional REST API. Uses the same saved pipeline as the Streamlit app (src.prediction).

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000      Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import secrets
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from src import config
from src.prediction import InvalidInput, ModelNotAvailable, load_bundle, model_status, predict, predict_batch

from .schemas import BatchRequest, BatchResponse, HealthResponse, PredictRequest, PredictResponse

log = logging.getLogger("verinews.api")

app = FastAPI(title="VeriNews-AI API", version=config.APP_VERSION,
              description="Classifies news text as FAKE or REAL with a TF-IDF + Logistic Regression model. "
                          "Outputs are model estimates, not fact-checks.")


def require_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = config.api_key()
    if expected and not (x_api_key and secrets.compare_digest(x_api_key.encode(), expected.encode())):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


@app.exception_handler(ModelNotAvailable)
async def _no_model(_: Request, exc: ModelNotAvailable):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(InvalidInput)
async def _bad_input(_: Request, exc: InvalidInput):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def _unexpected(_: Request, exc: Exception):
    log.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/health", response_model=HealthResponse)
def health():
    st = model_status()
    meta = st["metadata"] or {}
    return {"status": "ok", "app_version": config.APP_VERSION,
            "model": {"state": st["state"], "message": st["message"], "model_version": meta.get("model_version"),
                      "trained_at": meta.get("trained_at"), "is_demo": meta.get("is_demo")}}


@app.get("/metrics", dependencies=[Depends(require_key)])
def metrics():
    bundle = load_bundle()
    if not bundle.metrics:
        raise HTTPException(status_code=404, detail="No saved evaluation metrics found.")
    return bundle.metrics


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_key)])
def predict_one(req: PredictRequest):
    return predict(load_bundle(), req.text, req.headline)


@app.post("/predict/batch", response_model=BatchResponse, dependencies=[Depends(require_key)])
def predict_many(req: BatchRequest):
    bundle = load_bundle()
    return {"model_version": bundle.version, "is_demo": bundle.is_demo, "calibrated": False,
            "results": predict_batch(bundle, req.texts)}
