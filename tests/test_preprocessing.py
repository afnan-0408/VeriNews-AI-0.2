import pandas as pd
import pytest

from src.data_processing import (DatasetError, find_leakage, normalize_labels, prepare_dataset, read_csv_safe)

LONG = "This is a sufficiently long article body used for testing the cleaning logic number {}."


def test_label_normalization_and_unknown():
    out, unknown = normalize_labels(["Fake", "TRUE", " real ", "banana", None, "1"])
    assert out == ["FAKE", "REAL", "REAL", None, None, None]
    assert unknown == ["1", "banana"]          # numeric labels are never guessed


def test_custom_label_map_and_bad_target():
    out, _ = normalize_labels(["1", "0"], {"1": "FAKE", "0": "REAL"})
    assert out == ["FAKE", "REAL"]
    with pytest.raises(DatasetError):
        normalize_labels(["x"], {"x": "MAYBE"})


def test_cleaning_missing_duplicates_conflicts_unknown():
    df = pd.DataFrame({
        "text": [LONG.format(1), LONG.format(1), None, "short", LONG.format(2), LONG.format(2), LONG.format(3), LONG.format(4)],
        "label": ["fake", "fake", "fake", "fake", "fake", "real", "mystery", "real"],
    })
    clean, rep = prepare_dataset(df, "text", "label")
    assert rep.rows_before == 8
    assert rep.dropped_missing_text == 1 and rep.dropped_too_short == 1
    assert rep.dropped_duplicates == 1            # second copy of article 1
    assert rep.dropped_label_conflicts == 2       # article 2 appears as both fake and real
    assert rep.dropped_unknown_label == 1 and rep.unknown_label_values == ["mystery"]
    assert rep.rows_after == len(clean) == 2
    assert set(clean["label"]) == {"FAKE", "REAL"}


def test_invalid_datasets():
    with pytest.raises(DatasetError):
        prepare_dataset(pd.DataFrame(), "text", "label")
    with pytest.raises(DatasetError):
        prepare_dataset(pd.DataFrame({"a": [1]}), "text", "label")
    with pytest.raises(DatasetError):   # only one class
        prepare_dataset(pd.DataFrame({"text": [LONG.format(1), LONG.format(2)], "label": ["fake", "fake"]}), "text", "label")


def test_agency_prefix_strip_is_optional():
    df = pd.DataFrame({"text": ["WASHINGTON (Reuters) - " + LONG.format(1), LONG.format(2)], "label": ["real", "fake"]})
    kept, _ = prepare_dataset(df, "text", "label")
    stripped, _ = prepare_dataset(df, "text", "label", strip_agency_prefix=True)
    assert kept["text"].iloc[0].startswith("WASHINGTON")
    assert not stripped["text"].iloc[0].startswith("WASHINGTON")


def test_read_csv_safe(tmp_path):
    p = tmp_path / "a.csv"
    p.write_bytes("text,label\nhéllo wörld,fake\n".encode("cp1252"))
    assert len(read_csv_safe(p)) == 1
    with pytest.raises(DatasetError):
        read_csv_safe(tmp_path / "missing.csv")
    (tmp_path / "e.csv").write_text("")
    with pytest.raises(DatasetError):
        read_csv_safe(tmp_path / "e.csv")
    (tmp_path / "x.exe").write_text("text,label\na,b\n")
    with pytest.raises(DatasetError):
        read_csv_safe(tmp_path / "x.exe")
    with pytest.raises(DatasetError):
        read_csv_safe(b"text,label\n" + b"a,b\n" * 1000, max_mb=0)


def test_find_leakage_detects_exact_and_near():
    train = ["the quick brown fox jumps over the lazy dog today", "completely different sentence about finance markets"]
    test = ["The quick brown fox jumps over the lazy dog today", "an unrelated article regarding sport results"]
    exact, near = find_leakage(train, test)
    assert exact.tolist() == [True, False] and near.tolist() == [True, False]
