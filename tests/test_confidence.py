import sqlite3

from querycraft.confidence import (auroc, selective, signature, thresholds_for_coverage, thresholds_for_targets,
                                   vote)


def _db(tmp_path):
    p = tmp_path / "t.sqlite"
    con = sqlite3.connect(p)
    con.executescript("CREATE TABLE t(a INT, b TEXT); INSERT INTO t VALUES (1,'x'),(2,'y'),(3,'x');")
    con.commit()
    con.close()
    return str(p)


def test_signature_ignores_row_order_keeps_duplicates():
    assert signature([(1,), (2,)]) == signature([(2,), (1,)])
    assert signature([(1,), (1,)]) != signature([(1,)])


def test_vote_majority_by_result_not_by_text(tmp_path):
    db = _db(tmp_path)
    same_rows = ["SELECT a FROM t WHERE b='x'", "SELECT a FROM t WHERE b = 'x' ORDER BY a DESC",
                 "select a from t where b='x'"]
    other = ["SELECT a FROM t", "SELECT nope FROM t"]  # one different result, one failure (no vote)
    sql, conf, n_results = vote({"db_path": db, "samples": same_rows + other, "greedy": "SELECT a FROM t"})
    assert sql == same_rows[0] and conf == 3 / 5 and n_results == 2


def test_vote_tie_prefers_greedy_and_all_failures_fall_back(tmp_path):
    db = _db(tmp_path)
    sql, conf, _ = vote({"db_path": db, "samples": ["SELECT a FROM t", "SELECT b FROM t"], "greedy": "SELECT b FROM t"})
    assert sql == "SELECT b FROM t" and conf == 0.5
    sql, conf, n = vote({"db_path": db, "samples": ["SELECT x", "bad sql"], "greedy": "SELECT 1"})
    assert sql == "SELECT 1" and conf == 0.0 and n == 0


def test_selective_metrics():
    conf = [1.0, 0.875, 0.5, 0.25]
    correct = [True, True, False, False]
    assert auroc(conf, correct) == 1.0
    assert selective(conf, correct, 0.875) == {"threshold": 0.875, "coverage": 50.0, "accuracy": 100.0}
    assert thresholds_for_targets(conf, correct, (0.6, 0.99)) == {"0.6": 0.5, "0.99": 0.875}
    # answer at least half: 0.875 keeps 2 of 4; at least 70%: needs 3 of 4, so 0.5
    assert thresholds_for_coverage(conf, (0.5, 0.7)) == {"0.5": 0.875, "0.7": 0.5}
