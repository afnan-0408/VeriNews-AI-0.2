import pytest

from src import database as db


def test_insert_query_delete_and_privacy(tmp_path):
    p = tmp_path / "t.sqlite3"
    a = db.add_prediction("FAKE", 0.1, "secret article text here", "Head", "src-a", "v1", 3.2, store_text=False, db_path=p)
    b = db.add_prediction("REAL", 0.9, "other article words", "Head2", "src-b", "v1", 1.0, store_text=True, db_path=p)
    rows = {r["id"]: r for r in db.query_history(db_path=p)}
    assert rows[a]["text"] is None and rows[a]["headline"] is None and rows[a]["text_stored"] == 0
    assert rows[b]["text"] == "other article words" and rows[b]["headline"] == "Head2"
    assert db.count(p) == 2
    assert [r["id"] for r in db.query_history(labels=["REAL"], db_path=p)] == [b]
    assert [r["id"] for r in db.query_history(search="src-a", db_path=p)] == [a]
    assert db.query_history(search="100%_", db_path=p) == []          # wildcard characters are escaped
    assert db.delete_ids([a], db_path=p) == 1 and db.count(p) == 1
    assert db.delete_all(db_path=p) == 1 and db.count(p) == 0


def test_add_many_and_invalid_label_rolls_back(tmp_path):
    p = tmp_path / "t.sqlite3"
    assert db.add_many([{"label": "FAKE", "score_real": 0.2, "text": "a b c"}] * 3, db_path=p) == 3
    with pytest.raises(db.DatabaseError):
        db.add_many([{"label": "REAL", "score_real": 0.8, "text": "x"}, {"label": "MAYBE", "score_real": 0.5, "text": "y"}], db_path=p)
    assert db.count(p) == 3                                           # failed batch left nothing behind


def test_date_filter(tmp_path):
    p = tmp_path / "t.sqlite3"
    db.add_prediction("FAKE", 0.1, "t t t", db_path=p)
    assert len(db.query_history(date_from="2000-01-01", date_to="2999-01-01", db_path=p)) == 1
    assert db.query_history(date_to="2000-01-01", db_path=p) == []
