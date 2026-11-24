"""v2 pieces that need no GPU: abstain metrics, drift report, gate replay scoring, retriever table picking."""
import json
import sqlite3
import time

import pytest

from querycraft.abstain import at_coverage, brier, compare, ece, sign_p
from querycraft.gate import result_hashes, score, sha
from querycraft.retriever import pick, table_text


def test_brier_and_ece():
    assert brier([1.0, 0.0], [1, 0]) == 0.0
    assert brier([0.5, 0.5], [1, 0]) == 0.25
    assert ece([0.9] * 10, [1] * 9 + [0]) == 0.0           # 90% confident, 90% right
    assert ece([0.9] * 10, [1] * 5 + [0] * 5) == 0.4


def test_at_coverage_answers_exactly_that_share_highest_first():
    s, y = [0.9, 0.1, 0.8, 0.2, 0.5], [1, 0, 1, 0, 0]
    assert at_coverage(s, y, 0.4) == {0, 2}
    assert len(at_coverage([0.5] * 10, [0] * 10, 0.7)) == 7  # ties: still exactly 70%


def test_compare_prefers_the_better_ranking():
    y = [1] * 50 + [0] * 50
    good = [1.0] * 50 + [0.0] * 50
    bad = [0.0] * 50 + [1.0] * 50
    c = compare(good, bad, y, 0.5, n_boot=50)
    assert c["learned_acc"] == 100.0 and c["baseline_acc"] == 0.0 and c["p"] < 0.001
    assert c["diff_ci95"][0] > 50
    assert sign_p(0, 0) == 1.0


def test_gate_replay_scores_cached_completions():
    rows = [{"id": "a", "sql": "SELECT 1 ORDER BY 1"}, {"id": "b", "sql": "SELECT 2"}, {"id": "c", "sql": "SELECT 3"}]
    prompts = ["p1", "p2", "p3"]
    cache = {sha("p1"): "SELECT 1 ORDER BY 1", sha("p2"): "```sql\nSELECT 9\n```"}  # p3 never cached -> wrong
    results = {sha("SELECT 1 ORDER BY 1"): {"error": None, **result_hashes([(1,)])},
               sha("SELECT 2"): {"error": None, **result_hashes([(2,)])},
               sha("SELECT 9"): {"error": None, **result_hashes([(9,)])},
               sha("SELECT 3"): {"error": None, **result_hashes([(3,)])}}
    m, per = score(rows, prompts, cache, results)
    assert m == {"n": 3, "ex": 33.33, "cache_hits": 2}
    assert [p["ex"] for p in per] == [True, False, False]


def test_result_hashes_follow_matching_rules():
    a, b = result_hashes([(1,), (2,)]), result_hashes([(2,), (1,)])
    assert a["multiset"] == b["multiset"] and a["ordered"] != b["ordered"]
    assert result_hashes([(1,)])["multiset"] != result_hashes([("1",)])["multiset"]  # types matter, as in SQLite


def test_pick_top_k_and_hop():
    tables = [{"name": "Orders", "fks": [("c", "Customers", "id")], "columns": []},
              {"name": "Customers", "fks": [], "columns": []}, {"name": "Other", "fks": [], "columns": []}]
    ranked = [["orders", 0.9], ["other", 0.5], ["customers", 0.1]]
    assert pick(ranked, tables, 1, False) == {"orders"}
    assert pick(ranked, tables, 1, True) == {"orders", "customers"}
    t = table_text({"name": "t", "columns": [{"name": "a", "type": "INT", "values": [1, 2]}]})
    assert t == "table t: a INT e.g. 1, 2"


def test_drift_flags_a_length_shift(tmp_path):
    pytest.importorskip("scipy")
    from querycraft.drift import report
    log = tmp_path / "q.sqlite"
    with sqlite3.connect(log) as c:
        c.execute("CREATE TABLE questions (id TEXT, ts REAL, db TEXT, question TEXT, hint TEXT, tables TEXT, sql TEXT, "
                  "confidence REAL, abstained INT, outcome TEXT, error TEXT, latency_s REAL, steps INT, tokens INT, model TEXT)")
        now = time.time()
        for i in range(60):
            c.execute("INSERT INTO questions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (str(i), now - 60, "shop", " ".join(["word"] * 40), "", "[]", "SELECT 1", 0.9, 0, "answered",
                       None, 1.0, 3, 100, "m"))
    ref = {"question_words": [10, 11, 12, 9, 10] * 20, "confidence": [0.5, 0.6] * 50, "known_dbs": {"bird"}}
    r = report(log, 7, ref=ref, now=now)
    assert "question_words" in r["flags"] and "confidence" in r["flags"]
    assert r["checks"]["database"]["share_on_unseen_databases"] == 1.0
    json.dumps(r)  # serialisable
