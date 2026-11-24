"""Agent tools and the loop's control flow on a toy database with a scripted model (CPU only, no GPU)."""
import sqlite3

from querycraft.agent import describe_table, literal_columns, run_batch, sample_values
from querycraft.schema import read_schema


def _db(tmp_path):
    p = tmp_path / "shop.sqlite"
    con = sqlite3.connect(p)
    con.executescript("CREATE TABLE city(id INTEGER PRIMARY KEY, name TEXT);"
                      "CREATE TABLE store(id INTEGER PRIMARY KEY, city_id INT REFERENCES city(id), sales INT);"
                      "INSERT INTO city VALUES (1,'New York'),(2,'Boston');"
                      "INSERT INTO store VALUES (1,1,10),(2,1,5),(3,2,7);")
    con.commit()
    con.close()
    return str(p)


class Scripted:
    """Returns queued answers in order; counts tokens as 1 + 1 per call."""
    tok = None

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def complete(self, prompts, temperature=0.0, seed=None):
        self.prompts += prompts
        return [(self.answers.pop(0), 1, 1) for _ in prompts]


class Tok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "\n".join(m["content"] for m in msgs)


OPTS = {"max_steps": 8, "max_repairs": 2, "retry_temperature": 0.6, "repair_on_empty": True, "widen_on_error": True}


def _ex(db, q="total sales in New York"):
    return {"id": "x1", "db_id": "shop", "db_path": db, "question": q, "evidence": "", "sql": "SELECT 1"}


def test_describe_and_sample(tmp_path):
    db = _db(tmp_path)
    assert "CREATE TABLE city" in describe_table(db, "CITY")
    assert sample_values(db, "city", "name", like="york")[0] == "New York"


def test_literal_columns_resolves_aliases(tmp_path):
    db = _db(tmp_path)
    sql = "SELECT SUM(s.sales) FROM store s JOIN city c ON s.city_id = c.id WHERE c.name = 'new york'"
    assert literal_columns(sql, read_schema(db)) == [("city", "name", "new york")]


def test_loop_repairs_error_then_empty_then_finishes(tmp_path):
    db = _db(tmp_path)
    llm = Scripted(["SELECT nope FROM store",                                   # error -> describe + retry
                    "SELECT SUM(sales) FROM store JOIN city ON city_id = city.id WHERE name = 'new york'",  # empty
                    "SELECT SUM(sales) FROM store JOIN city ON city_id = city.id WHERE name = 'New York'"])
    llm.tok = Tok()
    [r] = run_batch(llm, [_ex(db)], lambda ex: {"store", "city"}, OPTS)
    assert r["pred_sql"].endswith("'New York'") and r["repairs"] == 2
    tools = [t["tool"] for t in r["trace"]]
    assert tools[0] == "retrieve_tables" and tools[-1] == "finish"
    assert "describe_table" in tools and "sample_values" in tools
    assert "no such column" in llm.prompts[1] and "'New York'" in llm.prompts[2]  # the model saw the observations
    assert r["steps"] <= OPTS["max_steps"]


def test_step_limit_and_fallback_to_last_working_query(tmp_path):
    db = _db(tmp_path)
    llm = Scripted(["SELECT sales FROM store WHERE id = 99", "SELECT nope", "SELECT nope2"])
    llm.tok = Tok()
    [r] = run_batch(llm, [_ex(db)], lambda ex: {"store"}, OPTS)
    assert r["pred_sql"] == "SELECT sales FROM store WHERE id = 99"  # ran without error, preferred over failures
    assert r["steps"] <= OPTS["max_steps"]
