import sqlite3

import pytest

from querycraft.reward import binary_reward, shaped_one, shaped_reward

GOLD = "SELECT u.name, COUNT(*) FROM u JOIN o ON u.id = o.uid GROUP BY u.name"


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "t.sqlite"
    con = sqlite3.connect(p)
    con.executescript("CREATE TABLE u(id INT, name TEXT); CREATE TABLE o(id INT, uid INT, amt INT);"
                      "INSERT INTO u VALUES (1,'a'),(2,'b'); INSERT INTO o VALUES (1,1,5),(2,1,7),(3,2,1);")
    con.commit()
    con.close()
    return str(p)


@pytest.mark.parametrize("pred, want", [
    (GOLD, 1.0),
    ("SELECT name, n FROM (SELECT u.name, COUNT(*) n FROM u JOIN o ON u.id=o.uid GROUP BY u.name)", 1.0),  # same rows
    ("SELECT u.name, SUM(o.amt) FROM u JOIN o ON u.id = o.uid GROUP BY u.name", 0.5),  # wrong aggregation (and columns)
    ("SELECT u.name, COUNT(*) FROM u JOIN o ON u.id = o.id GROUP BY u.name", 0.4),  # wrong join (and columns)
    ("SELECT name FROM u", 0.1),  # runs, nothing matches
    ("SELECT nope FROM u", 0.0),  # fails, not a syntax error
    ("SELEC name FROM u", -0.2),  # syntax error
    ("", -0.2),  # empty output counts as a syntax error, so saying nothing is never a safe policy
])
def test_shaped_table(db, pred, want):
    assert shaped_one(db, GOLD, pred) == pytest.approx(want)


def test_wrong_never_reaches_correct(db):
    # best possible wrong answer: identical skeleton, different literal
    gold = "SELECT u.name FROM u JOIN o ON u.id = o.uid WHERE o.amt > 4"
    assert shaped_one(db, gold, "SELECT u.name FROM u JOIN o ON u.id = o.uid WHERE o.amt > 6") == pytest.approx(0.7)


def test_trl_signature_extracts_fenced_sql(db):
    comps = [f"```sql\n{GOLD};\n```", "SELECT name FROM u", [{"role": "assistant", "content": GOLD}]]
    kw = {"db_path": [db] * 3, "gold_sql": [GOLD] * 3}
    assert binary_reward(None, comps, **kw) == [1.0, 0.0, 1.0]
    assert shaped_reward(None, comps, **kw) == [1.0, 0.1, 1.0]
