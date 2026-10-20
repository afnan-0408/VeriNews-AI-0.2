import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

client = TestClient(app)


def test_health_without_model(monkeypatch, tmp_path):
    monkeypatch.setenv("VERINEWS_ARTIFACT_DIR", str(tmp_path))
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model"]["state"] == "missing"


def test_predict_without_model_is_503(monkeypatch, tmp_path):
    monkeypatch.setenv("VERINEWS_ARTIFACT_DIR", str(tmp_path))
    assert client.post("/predict", json={"text": "word " * 30}).status_code == 503


def test_health_and_metrics_with_model(model_env):
    assert client.get("/health").json()["model"]["state"] == "demo"
    assert "test" in client.get("/metrics").json()


def test_predict_valid_and_invalid(model_env, raw_df):
    row = raw_df.iloc[0]
    ok = client.post("/predict", json={"text": row.text, "headline": row.title})
    assert ok.status_code == 200 and ok.json()["label"] in ("FAKE", "REAL")
    assert client.post("/predict", json={"text": ""}).status_code == 422
    assert client.post("/predict", json={"text": "too short"}).status_code == 422
    assert client.post("/predict", json={"nope": 1}).status_code == 422
    assert client.post("/predict", content=b"{bad json", headers={"content-type": "application/json"}).status_code == 422


def test_batch(model_env, raw_df):
    r = client.post("/predict/batch", json={"texts": [raw_df.iloc[0].text, "short"]})
    assert r.status_code == 200
    items = r.json()["results"]
    assert items[0]["status"] == "ok" and items[1]["status"].startswith("skipped")
    assert client.post("/predict/batch", json={"texts": []}).status_code == 422


def test_api_key_enforced_when_configured(model_env, monkeypatch):
    monkeypatch.setenv("VERINEWS_API_KEY", "secret-test-key")
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"X-API-Key": "secret-test-key"}).status_code == 200
    assert client.get("/health").status_code == 200
