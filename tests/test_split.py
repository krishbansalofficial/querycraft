import pytest

from querycraft import data


def fake_train(n_db=30, per_db=40):
    return [{"id": f"train_{d}_{i}", "db_id": f"db{d}", "question": "q", "sql": "SELECT 1"}
            for d in range(n_db) for i in range(per_db)]


def test_val_has_exactly_n_and_is_database_disjoint():
    tr, val, val_dbs = data.split_train(fake_train(), n_val=500, seed=0)
    assert len(val) == 500
    assert {r["db_id"] for r in val} == set(val_dbs)
    assert not {r["db_id"] for r in tr} & set(val_dbs)


def test_split_is_deterministic():
    a = data.split_train(fake_train(), 500, seed=1)
    b = data.split_train(fake_train(), 500, seed=1)
    assert [r["id"] for r in a[1]] == [r["id"] for r in b[1]]


def test_leakage_check_catches_dev_db_in_train_or_val():
    dev = [{"db_id": "dev_db"}]
    with pytest.raises(AssertionError):
        data.check_no_leakage({"train": [{"db_id": "dev_db"}], "val": [], "dev": dev})
    with pytest.raises(AssertionError):
        data.check_no_leakage({"train": [], "val": [{"db_id": "dev_db"}], "dev": dev})
    with pytest.raises(AssertionError):
        data.check_no_leakage({"train": [{"db_id": "x"}], "val": [{"db_id": "x"}], "dev": dev})
    data.check_no_leakage({"train": [{"db_id": "a"}], "val": [{"db_id": "b"}], "dev": dev})


@pytest.mark.skipif(not (data.PROC / "train.jsonl").exists(), reason="processed BIRD splits not built")
def test_real_splits_have_no_dev_database_in_train_or_val():
    s = {k: data.load_split(k) for k in ("train", "val", "dev", "heldback")}
    dev_dbs = {r["db_id"] for r in s["dev"]} | {r["db_id"] for r in s["heldback"]}
    assert not dev_dbs & {r["db_id"] for r in s["train"]}
    assert not dev_dbs & {r["db_id"] for r in s["val"]}
    assert not {r["db_id"] for r in s["train"]} & {r["db_id"] for r in s["val"]}
    assert len(s["val"]) == 500 and len(s["dev"]) == 1534
