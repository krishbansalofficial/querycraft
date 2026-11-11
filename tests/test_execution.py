import sqlite3

import pytest

from querycraft.execution import execute, has_top_level_order_by, results_match, score_one, tag_error
from querycraft.prompts import extract_sql


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    p = tmp_path_factory.mktemp("db") / "t.sqlite"
    c = sqlite3.connect(p)
    c.executescript("""
        CREATE TABLE team(id INTEGER PRIMARY KEY, name TEXT, city TEXT);
        CREATE TABLE player(id INTEGER PRIMARY KEY, name TEXT, team_id INTEGER REFERENCES team(id), goals INTEGER);
        INSERT INTO team VALUES (1,'Reds','Leeds'),(2,'Blues','York');
        INSERT INTO player VALUES (1,'Ann',1,5),(2,'Bob',1,3),(3,'Cy',2,7),(4,'Di',2,3);
    """)
    c.commit()
    c.close()
    return str(p)


def item(db, pred, gold):
    return {"db_path": db, "pred_sql": pred, "gold_sql": gold}


def test_order_insensitive_without_order_by(db):
    r = score_one(item(db, "SELECT name FROM player ORDER BY name DESC", "SELECT name FROM player"))
    assert r["ex"] and r["ex_bird"]


def test_order_sensitive_with_order_by(db):
    gold = "SELECT name FROM player ORDER BY goals DESC, name"
    assert score_one(item(db, gold, gold))["ex"]
    r = score_one(item(db, "SELECT name FROM player ORDER BY name", gold))
    assert not r["ex"] and r["ex_bird"]  # same set, wrong order: BIRD metric passes, ours does not


def test_subquery_order_by_is_not_top_level():
    assert not has_top_level_order_by("SELECT * FROM (SELECT a FROM t ORDER BY a) LIMIT 5")
    assert has_top_level_order_by("SELECT a FROM t ORDER BY a")


def test_duplicates_matter_for_ex_not_bird():
    assert not results_match([(1,), (1,)], [(1,)], ordered=False)
    assert results_match([(2,), (1,)], [(1,), (2,)], ordered=False)


def test_timeout(db):
    slow = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT count(*) FROM c"
    rows, err = execute(db, slow, timeout=0.5)
    assert rows is None and err == "timeout"
    assert score_one(item(db, slow, "SELECT 1"), timeout=0.5)["error_tag"] == "timeout"


def test_read_only(db):
    _, err = execute(db, "DELETE FROM player")
    assert err is not None and "readonly" in err.replace(" ", "").lower()
    assert execute(db, "SELECT count(*) FROM player")[0] == [(4,)]


@pytest.mark.parametrize("pred,tag", [
    ("SELEC name FROM player", "syntax"),
    ("", "syntax"),
    ("SELECT name FROM players", "wrong_table"),
    ("SELECT nme FROM player", "wrong_column"),
    ("SELECT name FROM team", "wrong_table"),
    ("SELECT p.name FROM player p JOIN team t ON p.id = t.id WHERE t.city='Leeds'", "wrong_join"),
    ("SELECT p.name FROM player p JOIN team t ON p.team_id = t.id WHERE t.city='York'", "other"),
    ("SELECT max(goals) FROM player", "wrong_aggregation"),
    ("SELECT goals FROM player WHERE team_id = 1", "wrong_column"),
    ("SELECT name FROM player WHERE team_id = 2", "other"),
])
def test_error_tags(db, pred, tag):
    gold = "SELECT p.name FROM player p JOIN team t ON p.team_id = t.id WHERE t.city='Leeds'" if "JOIN" in pred \
        else "SELECT name FROM player WHERE team_id = 1"
    r = score_one(item(db, pred, gold))
    assert not r["ex"] and r["error_tag"] == tag, r


def test_wrong_join_tag():
    assert tag_error("SELECT a.x FROM a JOIN b ON a.id=b.id", "SELECT a.x FROM a JOIN b ON a.id=b.id JOIN b AS b2 ON 1", None) == "wrong_join"


def test_extract_sql():
    assert extract_sql("```sql\nSELECT 1;\n```\nexplanation") == "SELECT 1"
    assert extract_sql("SELECT a FROM t; SELECT 2") == "SELECT a FROM t"
    assert extract_sql("  SELECT 'x'  ") == "SELECT 'x'"
