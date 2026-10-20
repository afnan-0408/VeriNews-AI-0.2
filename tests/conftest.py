import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_processing import prepare_dataset  # noqa: E402
from src.demo_data import make_demo_dataframe  # noqa: E402
from src.training import TrainConfig, train_model  # noqa: E402


@pytest.fixture(scope="session")
def raw_df():
    return make_demo_dataframe(n_per_class=60, seed=11)


@pytest.fixture(scope="session")
def trained_dir(tmp_path_factory, raw_df):
    d = tmp_path_factory.mktemp("artifacts")
    clean, report = prepare_dataset(raw_df, "text", "label", "title")
    train_model(clean, TrainConfig(min_df=1, cv_folds=3), d, {"name": "synthetic test data", "uses_title": True},
                report.to_dict(), is_demo=True)
    return d


@pytest.fixture()
def model_env(monkeypatch, trained_dir, tmp_path):
    monkeypatch.setenv("VERINEWS_ARTIFACT_DIR", str(trained_dir))
    monkeypatch.setenv("VERINEWS_DB_PATH", str(tmp_path / "h.sqlite3"))
    return trained_dir
