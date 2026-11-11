from types import SimpleNamespace

import pytest

pytest.importorskip("transformers")
from querycraft.train import PruneCheckpoints  # noqa: E402


def test_prune_keeps_candidates_and_newest_optimizer(tmp_path):
    for s in (70, 140, 280, 350):
        d = tmp_path / f"checkpoint-{s}"
        d.mkdir()
        (d / "optimizer.pt").write_text("x")
        (d / "adapter_model.safetensors").write_text("w")
    PruneCheckpoints([280, 420, 560]).on_save(SimpleNamespace(output_dir=str(tmp_path)),
                                              SimpleNamespace(is_world_process_zero=True), None)
    left = sorted(p.name for p in tmp_path.iterdir())
    assert left == ["checkpoint-280", "checkpoint-350"]
    assert not (tmp_path / "checkpoint-280" / "optimizer.pt").exists()  # candidate: weights only
    assert (tmp_path / "checkpoint-280" / "adapter_model.safetensors").exists()
    assert (tmp_path / "checkpoint-350" / "optimizer.pt").exists()  # newest: still resumable
