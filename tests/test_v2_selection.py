"""v2 rule: every choice (checkpoint, threshold, k, prompt, hyperparameter) is made on val. Dev and held back are
for final reporting only. Two checks: a runtime guard (load_split refuses dev/held back inside @selection code)
and a static one (no @selection function in a v2 module names those splits). Fails if either is bypassed."""
import ast
import re
from pathlib import Path

import pytest

from querycraft import data

V2_MODULES = ["retriever", "abstain", "agent", "gate", "bench", "drift"]  # v2 modules that make choices
ENTRY_POINTS = {"retriever": ["stage_sweep"], "abstain": ["fit"], "agent": ["tune"]}
FORBIDDEN = re.compile(r"\b(dev|heldback|held_back|iowa)\b")


def _files():
    return [p for m in V2_MODULES if (p := Path("querycraft") / f"{m}.py").exists()]


def test_guard_refuses_report_splits():
    @data.selection
    def choose(split):
        return data.load_split(split)

    for split in data.REPORT_ONLY:
        with pytest.raises(PermissionError):
            choose(split)


def test_guard_allows_val_and_releases(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PROC", tmp_path)
    for s in ("val", "dev"):
        (tmp_path / f"{s}.jsonl").write_text('{"id": 1}\n')

    @data.selection
    def choose():
        return data.load_split("val")

    assert choose() == [{"id": 1}]
    assert data.load_split("dev") == [{"id": 1}]  # reporting code outside a selection still reads dev


def _decorated(fn):
    return any((isinstance(d, ast.Name) and d.id == "selection") or (isinstance(d, ast.Attribute) and d.attr == "selection")
               for d in fn.decorator_list)


def test_selection_code_never_names_report_splits():
    bad = []
    for f in _files():
        for fn in ast.walk(ast.parse(f.read_text())):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and _decorated(fn):
                for node in ast.walk(fn):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str) and FORBIDDEN.search(node.value):
                        bad.append(f"{f}:{node.lineno} {fn.name}: {node.value!r}")
    assert not bad, "selection code names a report only split:\n" + "\n".join(bad)


def test_entry_points_are_marked():
    found = 0
    for mod, names in ENTRY_POINTS.items():
        p = Path("querycraft") / f"{mod}.py"
        if not p.exists():
            continue
        fns = {fn.name: fn for fn in ast.walk(ast.parse(p.read_text())) if isinstance(fn, ast.FunctionDef)}
        for n in names:
            assert n in fns and _decorated(fns[n]), f"{mod}.{n} makes a choice and must be @selection"
            found += 1
    assert found >= 1
