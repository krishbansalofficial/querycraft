"""Validate the 30 Iowa Liquor questions by execution and write them as the `iowa` split (a second held back set).

    python scripts/iowa_questions.py      # -> data/processed/iowa.jsonl; fails if any gold query errors or is empty

Every gold query must run and return at least one non NULL value. That checks the SQL runs, not that the answer
is right: the file stays marked "needs human review" until a person has checked it.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from querycraft.agent import is_empty  # noqa: E402
from querycraft.execution import execute  # noqa: E402

DB = "data/warehouse/iowa_liquor.sqlite"
Q = Path("data/warehouse/iowa_questions.json")
OUT = Path("data/processed/iowa.jsonl")


def main():
    spec = json.loads(Q.read_text())
    rows, bad = [], []
    for q in spec["questions"]:
        res, err = execute(DB, q["sql"], 120.0)
        ok = err is None and not is_empty(res)
        print(f"{q['id']:>2} {'ok ' if ok else 'BAD'} {len(res or [])} rows  {str((res or [err])[:2])[:90]}")
        if not ok:
            bad.append(q["id"])
        rows.append({"id": f"iowa_{q['id']}", "db_id": "iowa_liquor", "db_path": DB, "question": q["question"],
                     "evidence": q["evidence"], "sql": q["sql"], "difficulty": q["difficulty"]})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows) - len(bad)}/{len(rows)} gold queries return rows; status: {spec['status']}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
