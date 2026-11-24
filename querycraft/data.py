"""BIRD loading and splits.

Splits (written to data/processed/*.jsonl):
  train      BIRD train minus the validation databases
  val        500 pairs from whole BIRD train databases held out of training. Holding out by database
             (not by row) mirrors BIRD dev, whose databases are unseen at training time.
             All model selection (checkpoints, prompts) happens here.
  dev        BIRD dev (1,534). Held out test set. Never used for selection.
  heldback   ~100 questions written for this project on dev databases; final scoring only.
"""
import contextvars
import functools
import json
import random
import sys
from collections import Counter
from pathlib import Path

from .config import DATA_DIR

RAW, PROC = DATA_DIR / "raw", DATA_DIR / "processed"
HELDBACK = Path("data/heldback/heldback.json")  # tracked in git, not under DATA_DIR


def _find(pattern):
    hits = sorted(h for h in RAW.rglob(pattern) if "__MACOSX" not in h.parts)  # skip macOS resource forks
    if not hits:
        raise FileNotFoundError(f"{pattern} not found under {RAW}; run `make data`")
    return hits[0]


def _db_path(db_root: Path, db_id: str) -> str:
    return str(db_root / db_id / f"{db_id}.sqlite")


def load_raw():
    train_json, dev_json = _find("train.json"), _find("dev.json")
    train_db = _find("train_databases")
    dev_db = _find("dev_databases")
    train = [{"id": f"train_{i}", "db_id": r["db_id"], "db_path": _db_path(train_db, r["db_id"]),
              "question": r["question"], "evidence": r.get("evidence", ""), "sql": r["SQL"], "difficulty": None}
             for i, r in enumerate(json.loads(train_json.read_text()))]
    dev = [{"id": f"dev_{r['question_id']}", "db_id": r["db_id"], "db_path": _db_path(dev_db, r["db_id"]),
            "question": r["question"], "evidence": r.get("evidence", ""), "sql": r["SQL"], "difficulty": r["difficulty"]}
           for r in json.loads(dev_json.read_text())]
    return train, dev, dev_db


def split_train(train, n_val=500, seed=0):
    """Pick random whole databases until they cover n_val pairs; take exactly n_val of their pairs
    for validation and drop the rest so no validation database is ever trained on."""
    by_db = {}
    for r in train:
        by_db.setdefault(r["db_id"], []).append(r)
    dbs = sorted(by_db)
    random.Random(seed).shuffle(dbs)
    val_dbs, n = [], 0
    for db in dbs:
        if n >= n_val:
            break
        val_dbs.append(db)
        n += len(by_db[db])
    pool = [r for db in val_dbs for r in by_db[db]]
    random.Random(seed).shuffle(pool)
    val = sorted(pool[:n_val], key=lambda r: r["id"])
    tr = [r for r in train if r["db_id"] not in set(val_dbs)]
    return tr, val, val_dbs


def check_no_leakage(splits: dict):
    """Raise if any database appears in two of {train, val} vs {dev, heldback}, or train vs val."""
    dbs = {k: {r["db_id"] for r in v} for k, v in splits.items()}
    test_dbs = dbs.get("dev", set()) | dbs.get("heldback", set())
    for k in ("train", "val"):
        bad = dbs.get(k, set()) & test_dbs
        if bad:
            raise AssertionError(f"leakage: {k} shares databases with test sets: {sorted(bad)}")
    bad = dbs.get("train", set()) & dbs.get("val", set())
    if bad:
        raise AssertionError(f"leakage: train and val share databases: {sorted(bad)}")


def load_heldback(dev_db: Path):
    if not HELDBACK.exists():
        return []
    return [{"id": f"heldback_{r['id']}", "db_id": r["db_id"], "db_path": _db_path(dev_db, r["db_id"]),
             "question": r["question"], "evidence": r.get("evidence", ""), "sql": r["sql"],
             "difficulty": r.get("difficulty")} for r in json.loads(HELDBACK.read_text())]


REPORT_ONLY = ("dev", "heldback", "iowa")  # final reporting only: no choice may ever read them
_SELECTING = contextvars.ContextVar("selecting", default=False)


def selection(fn):
    """Mark code that makes a choice (checkpoint, threshold, k, prompt). While it runs, load_split refuses dev
    and held back; tests/test_v2_selection.py also checks every marked function never names those splits."""
    @functools.wraps(fn)
    def wrap(*a, **kw):
        tok = _SELECTING.set(True)
        try:
            return fn(*a, **kw)
        finally:
            _SELECTING.reset(tok)
    wrap.__selection__ = True
    return wrap


def load_split(name):
    if _SELECTING.get() and name in REPORT_ONLY:
        raise PermissionError(f"selection code may not read {name}: choices are made on val only")
    return [json.loads(l) for l in (PROC / f"{name}.jsonl").read_text().splitlines()]


def main(seed=0, n_val=500):
    train_all, dev, dev_db = load_raw()
    train, val, val_dbs = split_train(train_all, n_val, seed)
    splits = {"train": train, "val": val, "dev": dev, "heldback": load_heldback(dev_db)}
    check_no_leakage(splits)
    PROC.mkdir(parents=True, exist_ok=True)
    for k, v in splits.items():
        (PROC / f"{k}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in v))
    stats = {"bird_train_total": len(train_all), "bird_dev_total": len(dev), "seed": seed,
             "val_databases": val_dbs, "dropped_from_val_dbs": sum(r["db_id"] in val_dbs for r in train_all) - len(val),
             **{f"n_{k}": len(v) for k, v in splits.items()},
             **{f"n_db_{k}": len({r["db_id"] for r in v}) for k, v in splits.items()},
             "dev_difficulty": Counter(r["difficulty"] for r in dev)}
    (PROC / "split_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
