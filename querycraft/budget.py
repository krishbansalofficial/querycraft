"""Spend guardrail for free GPU credit: a job only starts if its worst case cost still fits under the cap,
so no run is ever cut off halfway by an empty balance (which would force a mid run move to the laptop).

Each rented config declares `budget_hours`: worst case A100 hours for train + eval. Once real jobs have run,
estimates are scaled up (never down) by how much slower reality was than the declared numbers.
"""
import json
from pathlib import Path

RATE_PER_H = 2.50   # Modal A100 80GB list price, USD
CAP_USD = 27.0      # $30 monthly free credit minus $3 for image build, data download and storage


def load(ledger_path):
    p = Path(ledger_path)
    return json.loads(p.read_text()) if p.exists() else []


def record(ledger_path, name, declared_h, actual_h):
    rows = load(ledger_path) + [{"name": name, "declared_h": declared_h, "actual_h": round(actual_h, 3),
                                 "usd": round(actual_h * RATE_PER_H, 2)}]
    Path(ledger_path).write_text(json.dumps(rows, indent=1))


def spent(rows):
    return sum(r["usd"] for r in rows)


def slowdown(rows):
    """How much slower jobs ran than declared (>= 1)."""
    d = sum(r["declared_h"] for r in rows)
    return max(1.0, sum(r["actual_h"] for r in rows) / d) if d else 1.0


def can_start(rows, declared_h, cap=CAP_USD):
    worst = declared_h * slowdown(rows) * RATE_PER_H
    return spent(rows) + worst <= cap, round(worst, 2)
