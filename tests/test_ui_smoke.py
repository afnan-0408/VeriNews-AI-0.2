"""Renders each Streamlit page headlessly and asserts there is no uncaught exception."""
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = ["app.py", "pages/1_News_Analyzer.py", "pages/2_Batch_Analysis.py", "pages/3_Model_Performance.py",
         "pages/4_Prediction_History.py", "pages/5_Dataset_Training.py", "pages/6_About.py"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_model(page, monkeypatch, tmp_path):
    monkeypatch.setenv("VERINEWS_ARTIFACT_DIR", str(tmp_path / "none"))
    monkeypatch.setenv("VERINEWS_DB_PATH", str(tmp_path / "h.sqlite3"))
    at = AppTest.from_file(page, default_timeout=30).run()
    assert not at.exception


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_with_model(page, model_env):
    at = AppTest.from_file(page, default_timeout=30).run()
    assert not at.exception
