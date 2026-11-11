import pytest

from querycraft import budget


def test_budget_blocks_job_that_could_overrun(tmp_path):
    led = tmp_path / "ledger.json"
    assert budget.can_start([], 10.0)[0]            # $25 worst case fits under $27
    assert not budget.can_start([], 11.0)[0]        # $27.50 does not
    budget.record(led, "a", declared_h=2.0, actual_h=3.0)   # ran 1.5x slower than declared
    rows = budget.load(led)
    assert budget.spent(rows) == 7.5
    ok, worst = budget.can_start(rows, 5.0)         # 5 h * 1.5 slowdown * $2.50 = $18.75; 7.5 + 18.75 <= 27
    assert ok and worst == 18.75
    assert not budget.can_start(rows, 6.0)[0]       # 7.5 + 22.5 > 27


def test_layout_guard_refuses_changed_batch(tmp_path):
    pytest.importorskip("transformers")
    from querycraft.train import check_layout
    t = {"method": "qlora", "rank": 16, "lr": 2e-4, "max_steps": 560, "per_device_batch": 4, "grad_accum": 4,
         "max_len": 2048}
    check_layout(tmp_path, t, 0)                    # first start writes layout.json
    check_layout(tmp_path, t, 0)                    # same layout resumes fine
    with pytest.raises(SystemExit, match="layout changed"):
        check_layout(tmp_path, t | {"per_device_batch": 1, "grad_accum": 16}, 0)
