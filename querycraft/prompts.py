"""Chat prompts for text to SQL. One builder is used for training targets, baselines and evaluation."""
import re
from functools import lru_cache

import sqlglot
from sqlglot import exp

from .schema import prune, read_schema, serialize

SYSTEM = ("You are an expert SQLite engineer. Given a database schema, optional external knowledge and a "
          "question, write one SQLite query that answers the question. Reply with only the SQL query.")


def user_message(schema_text, question, evidence):
    return (f"Database schema:\n{schema_text}\n\n"
            f"External knowledge: {evidence.strip() or 'None'}\n\n"
            f"Question: {question.strip()}")


class PromptBuilder:
    def __init__(self, tokenizer, schema_budget=1280, shots=()):
        self.tok = tokenizer
        self.budget = schema_budget
        self.shots = list(shots)  # train examples used as few shot turns

    @lru_cache(maxsize=200_000)
    def n_tokens(self, text):
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def schema_text(self, ex, pruned=True):
        tables = read_schema(ex["db_path"])
        if not pruned:
            return serialize(tables)
        keep, bare = prune(tables, ex["question"], ex.get("evidence", ""), self.n_tokens, self.budget)
        return serialize(tables, keep, bare)

    def _shot_schema(self, ex):
        # few shot examples only show the tables their gold SQL touches, to keep prompts short
        used = gold_tables(ex["sql"])
        tables = read_schema(ex["db_path"])
        keep = {t["name"]: {c["name"] for c in t["columns"]} for t in tables if t["name"].lower() in used}
        return serialize(tables, keep or None)

    def messages(self, ex, pruned=True):
        msgs = [{"role": "system", "content": SYSTEM}]
        for s in self.shots:
            msgs.append({"role": "user", "content": user_message(self._shot_schema(s), s["question"], s.get("evidence", ""))})
            msgs.append({"role": "assistant", "content": s["sql"]})
        msgs.append({"role": "user", "content": user_message(self.schema_text(ex, pruned), ex["question"], ex.get("evidence", ""))})
        return msgs

    def prompt_text(self, ex, pruned=True):
        return self.tok.apply_chat_template(self.messages(ex, pruned), tokenize=False, add_generation_prompt=True)


def gold_tables(sql):
    try:
        return {t.name.lower() for t in sqlglot.parse_one(sql, read="sqlite").find_all(exp.Table)}
    except Exception:
        return set()


_FENCE = re.compile(r"```(?:sql|sqlite)?\s*(.*?)(?:```|$)", re.S | re.I)


def extract_sql(text: str) -> str:
    """First fenced block if any, else raw text; keep only the first statement."""
    m = _FENCE.search(text)
    sql = m.group(1) if m else text
    # ponytail: naive split on ';' breaks if a string literal contains ';' (rare in BIRD); use sqlglot if it matters
    sql = next((s for s in sql.split(";") if s.strip()), "")
    return sql.strip()
