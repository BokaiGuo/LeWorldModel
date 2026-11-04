import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import run_reacher_pc_p1_fd_suffix_cuda_20261102_codexexclusive as launcher


def test_plan_binds_sources_and_rejects_root_reordering(tmp_path, monkeypatch):
    source = tmp_path / "preflight.json"
    source.write_text(json.dumps({
        "q_report_sha256": "q",
        "root_source_sha256": "root",
        "weights_sha256": "weights",
        "action_stats_sha256": "stats",
        "model_parameter_hash_expected": "params",
        "roots": [20263160, 20263161],
    }))
    q = tmp_path / "q.json"
    q.write_text(json.dumps({"rows": [{"seed": 20263160}, {"seed": 20263161}]}))
    monkeypatch.setattr(launcher.preflight, "Q_PATH", q)
    monkeypatch.setattr(launcher.preflight, "material", lambda: {})
    monkeypatch.setattr(launcher, "verify_preflight", lambda saved, actual: None)
    plan = launcher.plan(source)
    assert plan["schema"] == "pc_fd_suffix_cuda_development_v1"
    assert str(Path(launcher.search.__file__)) in plan["sources"]
    assert str(Path(launcher.strict.__file__)) in plan["sources"]
    assert plan["roots"] == [20263160, 20263161]
    q.write_text(json.dumps({"rows": [{"seed": 20263161}, {"seed": 20263160}]}))
    with pytest.raises(ValueError, match="root order"):
        launcher.plan(source)
