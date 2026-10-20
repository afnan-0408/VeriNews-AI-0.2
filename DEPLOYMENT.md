# Deployment

Nothing here has been deployed or verified live. Follow the steps and check each one.

## 1. Push to GitHub
```bash
git init && git add . && git commit -m "VeriNews-AI"
git branch -M main
git remote add origin https://github.com/<you>/VeriNews-AI.git      # create the empty repo on github.com first
git push -u origin main
```
`.gitignore` keeps datasets, models, the SQLite file and `.env` out of Git. Do not remove those entries.

## 2. Streamlit Community Cloud
1. Go to https://share.streamlit.io, sign in with GitHub, choose **Create app**.
2. Repository: `<you>/VeriNews-AI`, branch `main`, main file path `app.py`. Choose Python 3.11 or 3.12 under advanced settings.
3. Deploy. The first build installs `requirements.txt`; read the build log for conflicts (the UI packages are
   lower-bounded, not exact; after a good install, freeze versions).

## 3. Models and data on a hosted platform
Hosted file systems are **ephemeral**: files written at runtime (trained model, history database) disappear when the app
restarts or redeploys. Options:
* **Train inside the deployed app** on the Dataset & Training page after every restart (upload ISOT CSVs; keep
  `Max training rows` modest, free tiers have roughly 1 GB RAM).
* **Train locally, commit the model** by removing `artifacts/*` and `*.joblib` from `.gitignore` for that one file. The model
  is a few MB to tens of MB; check GitHub's 100 MB file limit. Then set `VERINEWS_MODEL_SHA256` (value from
  `artifacts/metadata.json`) as an app secret/env var so only that exact file loads.
* Prediction history is local to the instance and is lost on restart; treat it as a convenience, not a record.
* Never commit datasets: ISOT is large and has its own terms.

## 4. Environment variables / secrets
See `.env.example`. On Streamlit Cloud use **App settings -> Secrets** (or env vars). Never commit real keys.

## 5. API hosting
Any container/PaaS host works: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`. Provide the model artifacts to that
service too (same `src/` code). Set `VERINEWS_API_KEY`.
Rate limiting: do not expose the API publicly without it. Put it behind a reverse proxy or gateway that limits requests
per IP/key (e.g. nginx `limit_req`, Cloudflare rules), or add `slowapi`. Keep `VERINEWS_MAX_API_BATCH` small.

## 6. Resource notes
Training memory/time grow with rows and `max_features`. Defaults cap training at 50,000 rows and uploads at 50 MB;
lower `VERINEWS_MAX_TRAIN_ROWS` on small instances. Inference is cheap.
