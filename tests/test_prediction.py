import json

import pandas as pd
import pytest

from src import config
from src.data_processing import prepare_dataset
from src.prediction import (InvalidInput, ModelNotAvailable, load_bundle, model_status, predict, predict_batch,
                            validate_text)
from src.training import TrainConfig, TrainingError, train_model


def test_training_saves_real_artifacts(trained_dir):
    meta = json.loads((trained_dir / config.METADATA_FILENAME).read_text())
    metrics = json.loads((trained_dir / config.METRICS_FILENAME).read_text())
    assert (trained_dir / config.MODEL_FILENAME).is_file()
    assert meta["is_demo"] is True and meta["calibrated"] is False
    assert metrics["sizes"]["train"] > metrics["sizes"]["test"] > 0
    assert metrics["test"]["n_samples"] == metrics["sizes"]["test"]
    assert "TF-IDF fitted on the training split only" in metrics["evaluation_method"]


def test_model_loading_and_status(trained_dir):
    bundle = load_bundle(trained_dir)
    assert bundle.classes == ["FAKE", "REAL"]
    assert model_status(trained_dir)["state"] == "demo"


def test_missing_artifacts_do_not_crash(tmp_path):
    with pytest.raises(ModelNotAvailable):
        load_bundle(tmp_path)
    assert model_status(tmp_path)["state"] == "missing"


def test_tampered_model_is_rejected(trained_dir, tmp_path):
    import shutil
    d = tmp_path / "copy"
    shutil.copytree(trained_dir, d)
    with open(d / config.MODEL_FILENAME, "ab") as fh:
        fh.write(b"tamper")
    assert model_status(d)["state"] == "error"
    with pytest.raises(ModelNotAvailable):
        load_bundle(d)


@pytest.mark.parametrize("bad", ["", "   ", "too short", None, 123])
def test_invalid_input_rejected(trained_dir, bad):
    with pytest.raises(InvalidInput):
        predict(load_bundle(trained_dir), bad)


def test_valid_predictions_both_classes(trained_dir, raw_df):
    bundle = load_bundle(trained_dir)
    real = raw_df[raw_df.label == "REAL"].iloc[0]
    fake = raw_df[raw_df.label == "FAKE"].iloc[0]
    r, f = predict(bundle, real.text, real.title), predict(bundle, fake.text, fake.title)
    assert (r["label"], f["label"]) == ("REAL", "FAKE")
    assert abs(r["score_real"] + r["score_fake"] - 1) < 1e-9 and r["calibrated"] is False
    assert r["latency_ms"] >= 0 and r["explanation"]["known_terms"] > 0


def test_unknown_vocabulary_rejected(trained_dir):
    with pytest.raises(InvalidInput):
        predict(load_bundle(trained_dir), " ".join(["zzzqqq"] * 30))


def test_batch_csv_processing(trained_dir, raw_df, tmp_path):
    from src.data_processing import read_csv_safe
    p = tmp_path / "batch.csv"
    sample = raw_df.head(6)[["text"]].copy()
    sample.loc[len(sample)] = [""]
    sample.loc[len(sample)] = ["too short"]
    sample.to_csv(p, index=False)
    res = predict_batch(load_bundle(trained_dir), read_csv_safe(p)["text"].tolist())
    assert len(res) == 8
    assert [r["status"] == "ok" for r in res] == [True] * 6 + [False, False]
    assert all(r["label"] in ("FAKE", "REAL") for r in res[:6])


def test_training_rejects_bad_inputs(tmp_path):
    df = pd.DataFrame({"text": ["x " * 30] * 5, "label": ["FAKE"] * 5})
    with pytest.raises(TrainingError):
        train_model(df, TrainConfig(min_df=1), tmp_path)
    assert not (tmp_path / config.MODEL_FILENAME).exists()      # nothing saved on failure
    with pytest.raises(TrainingError):
        TrainConfig(test_size=0.9).validate()


def test_group_split_keeps_groups_apart(raw_df, tmp_path):
    df, _ = prepare_dataset(raw_df.assign(g=[i % 12 for i in range(len(raw_df))]), "text", "label", keep_cols=["g"])
    res = train_model(df, TrainConfig(min_df=1, split_strategy="group", group_col="g", cv_folds=0), tmp_path)
    assert res.metrics["cross_validation"] is None and res.metadata["split"]["strategy"] == "group"


def test_validate_text_limits():
    assert validate_text("word " * 20).startswith("word")
    with pytest.raises(InvalidInput):
        validate_text("a" * (config.MAX_TEXT_CHARS + 1), min_words=1)
