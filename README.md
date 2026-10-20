# VeriNews-AI

A transparent fake-news classifier: **TF-IDF + Logistic Regression** trained by you on a real dataset, evaluated on a
held-out test set, with a Streamlit interface, SQLite history and an optional FastAPI service.

> **Classification is not fact-checking.** The model scores writing patterns learned from its training data. Scores are
> uncalibrated. Nothing in this project verifies claims against evidence.

## Status of this build (read first)
* Built and tested in a sandbox **without network access**. `scikit-learn`, `pandas`, `numpy`, `joblib` and `sqlite3`
  were available, so the data, training, evaluation, prediction and database code were **run and tested** there.
* **Streamlit, Plotly, FastAPI and pytest could not be installed there**, so the Streamlit pages, the API and the
  `pytest` runner itself have **not been executed**. They are byte-compiled only. `tests/test_api.py` and
  `tests/test_ui_smoke.py` are written but unrun. Run `pytest` after installing (below) and fix anything it reports.
* No real dataset was available, so **no real model is shipped** and there are no real metrics yet. A synthetic demo
  dataset exists only to test the interface; models trained on it are flagged "demo" everywhere.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest                                                 # run the tests first
streamlit run app.py                                   # UI at http://localhost:8501
```
Python 3.11+ is required (the pinned NumPy needs it).

## Train a real model
1. Obtain the **ISOT Fake News Dataset** from its official source and check its licence. This project does not download
   datasets. Place `Fake.csv` and `True.csv` in `data/` (git-ignored).
2. Either open **Dataset & Training** in the app, or run:
```bash
python scripts/train_model.py --isot-dir data --strip-agency-prefix
python scripts/train_model.py --csv data/my.csv --text-col content --label-col verdict --fake-values false --real-values true
python scripts/train_model.py --demo        # SYNTHETIC data, interface testing only
```
Artifacts are written to `artifacts/` (`model.joblib`, `metadata.json`, `metrics.json`).

## REST API (optional)
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000        # OpenAPI docs: http://localhost:8000/docs
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"text": "...at least 15 words..."}'
```
Endpoints: `GET /health`, `GET /metrics`, `POST /predict`, `POST /predict/batch`. The API imports `src.prediction`, so it
uses exactly the same saved pipeline as the app. Set `VERINEWS_API_KEY` to require an `X-API-Key` header on everything
except `/health`.

## Design notes
* **No leakage**: split first; TF-IDF is inside the `Pipeline` and fitted on training data only. Exact and near-duplicate
  (cosine >= 0.9) test rows are detected and removed from the test set; counts are reported. Optional grouped and
  time-based splits.
* **No invented numbers**: every figure on the pages is read from `metrics.json` / `metadata.json`, written by training.
* **Explanations are real**: per-term `tf-idf x coefficient` contributions.
* **Labels**: unknown labels are dropped and counted, never defaulted to REAL. Numeric 0/1 labels must be mapped explicitly.
* **Artifact safety**: `joblib` files are pickles. The loader checks a SHA-256 recorded at training time and, if
  `VERINEWS_MODEL_SHA256` is set, a pinned hash. **Only load model files you trained yourself.** Uploaded CSVs are only
  parsed, never executed, and are not written to disk.
* **Privacy**: history stores metadata (time, label, score, word count, text hash, optional source). Full text and
  headline are stored only if you tick the opt-in box.

## Layout
`app.py` (Home), `pages/` (Streamlit pages), `src/` (config, data_processing, training, evaluation, prediction, database,
ui, demo_data), `api/` (FastAPI), `scripts/train_model.py`, `tests/`, `.streamlit/config.toml`.

Deployment: see [DEPLOYMENT.md](DEPLOYMENT.md). Environment variables: see `.env.example`.
