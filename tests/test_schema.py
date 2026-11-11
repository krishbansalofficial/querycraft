import sqlite3

from querycraft.schema import _cost, prune, read_schema, serialize


def make_db(path):
    c = sqlite3.connect(path)
    c.executescript("""
        CREATE TABLE driver(driverId INTEGER PRIMARY KEY, surname TEXT, nationality TEXT);
        CREATE TABLE lap(raceId INTEGER, driverId INTEGER REFERENCES driver(driverId), lap INTEGER, ms INTEGER,
                         PRIMARY KEY (raceId, driverId, lap));
    """ + "".join(f"CREATE TABLE junk{i}(id INTEGER PRIMARY KEY, {', '.join(f'filler_{j} TEXT' for j in range(15))});" for i in range(10)))
    c.execute("INSERT INTO driver VALUES (1, 'Hamilton', 'British')")
    c.commit()
    c.close()


def n_tokens(s):
    return len(s.split())


def test_composite_pk_and_values(tmp_path):
    p = tmp_path / "f1.sqlite"
    make_db(p)
    text = serialize(read_schema(str(p)))
    assert "PRIMARY KEY (raceId, driverId, lap)" in text
    assert "driverId INTEGER PRIMARY KEY, -- e.g. 1" in text
    assert "'Hamilton'" in text


def test_prune_fits_budget_and_keeps_relevant(tmp_path):
    p = tmp_path / "f1.sqlite"
    make_db(p)
    tables = read_schema(str(p))
    budget = 120
    assert _cost(tables, None, n_tokens) > budget
    keep, bare = prune(tables, "Fastest lap ms for driver Hamilton?", "", n_tokens, budget)
    assert _cost(tables, keep, n_tokens, bare) <= budget
    assert "driver" in keep and "lap" in keep
    assert "surname" in keep["driver"]
