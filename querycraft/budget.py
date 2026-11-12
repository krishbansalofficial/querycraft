"""Spend guardrail for free GPU credit: a job only starts if its worst case cost still fits under the cap,
so no run is ever cut off halfway by an empty balance (which would force a mid run move to the laptop).

Each rented config declares `budget_hours`: worst case A100 hours for train + eval. Once real jobs have run,
estimates are scaled up (never down) by how much slower reality was than the declared numbers.
Spend is counted per billing cycle (Modal's free $30 renews monthly from the signup day).
"""
import json
from datetime import date
from pathlib import Path

RATE_PER_H = 3.00   # Modal A100 80GB $2.50/h + 4 CPU cores $0.19/h + 32 GiB memory $0.26/h, rounded up
CAP_USD = 27.0      # $30 monthly free credit minus $3 for image build, data download and CPU jobs
CYCLE_DAY = 12      # signup day of month: the free credit renews on this day


def cycle_start(today=None):
    t = today or date.today()
    if t.day >= CYCLE_DAY:
        return date(t.year, t.month, CYCLE_DAY)
    return date(t.year - 1, 12, CYCLE_DAY) if t.month == 1 else date(t.year, t.month - 1, CYCLE_DAY)


def load(ledger_path):
    p = Path(ledger_path)
    return json.loads(p.read_text()) if p.exists() else []


def record(ledger_path, name, declared_h, actual_h, today=None):
    rows = load(ledger_path) + [{"name": name, "date": str(today or date.today()), "declared_h": declared_h,
                                 "actual_h": round(actual_h, 3), "usd": round(actual_h * RATE_PER_H, 2)}]
    Path(ledger_path).write_text(json.dumps(rows, indent=1))


def spent(rows, today=None):
    start = cycle_start(today)
    return round(sum(r["usd"] for r in rows if date.fromisoformat(r["date"]) >= start), 2)


def slowdown(rows):
    """How much slower jobs ran than declared (>= 1), over all history."""
    d = sum(r["declared_h"] for r in rows)
    return max(1.0, sum(r["actual_h"] for r in rows) / d) if d else 1.0


def can_start(rows, declared_h, cap=CAP_USD, today=None):
    worst = round(declared_h * slowdown(rows) * RATE_PER_H, 2)
    return spent(rows, today) + worst <= cap, worst
