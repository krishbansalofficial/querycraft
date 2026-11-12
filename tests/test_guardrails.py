from datetime import date

import pytest

from querycraft import budget

D = date(2026, 11, 20)


def test_budget_blocks_job_that_could_overrun(tmp_path):
    led = tmp_path / "ledger.json"
    assert budget.can_start([], 9.0, today=D)[0]          # 9 h * $3 = $27 fits the $27 cap
    assert not budget.can_start([], 9.5, today=D)[0]      # $28.50 does not
    budget.record(led, "a", declared_h=2.0, actual_h=3.0, today=D)   # ran 1.5x slower than declared
    rows = budget.load(led)
    assert budget.spent(rows, D) == 9.0
    ok, worst = budget.can_start(rows, 4.0, today=D)      # 4 h * 1.5 * $3 = $18; 9 + 18 <= 27
    assert ok and worst == 18.0
    assert not budget.can_start(rows, 4.1, today=D)[0]


def test_budget_resets_each_cycle(tmp_path):
    led = tmp_path / "ledger.json"
    budget.record(led, "a", 8.0, 8.0, today=date(2026, 11, 13))      # $24 in the Nov 12 cycle
    rows = budget.load(led)
    assert not budget.can_start(rows, 2.0, today=date(2026, 12, 11))[0]   # still Nov cycle
    assert budget.can_start(rows, 2.0, today=date(2026, 12, 12))[0]       # Dec cycle: fresh $27
    assert budget.cycle_start(date(2027, 1, 3)) == date(2026, 12, 12)


def test_layout_guard_refuses_changed_batch(tmp_path):
    pytest.importorskip("transformers")
    from querycraft.train import check_layout
    t = {"method": "qlora", "rank": 16, "lr": 2e-4, "max_steps": 560, "per_device_batch": 4, "grad_accum": 4,
         "max_len": 2048}
    check_layout(tmp_path, t, 0)                    # first start writes layout.json
    check_layout(tmp_path, t, 0)                    # same layout resumes fine
    with pytest.raises(SystemExit, match="layout changed"):
        check_layout(tmp_path, t | {"per_device_batch": 1, "grad_accum": 16}, 0)
