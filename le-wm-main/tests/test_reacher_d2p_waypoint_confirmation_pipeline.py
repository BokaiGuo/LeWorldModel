"""Orchestration-only mocks: ordering/sealing, never neural experiments."""
import json
import subprocess
import sys

import pytest

from experiments.spectral_reachability import run_reacher_d2p_waypoint_confirmation_cuda as pipeline


def configure(tmp_path, monkeypatch):
    policy = tmp_path / "policy"
    policy.mkdir()
    (policy / "weights.pt").write_bytes(b"unchanged fixture, not a model")
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({"status": "completed", "gates": {"fixture": True}, "rows": [{"source_episode": 100}],
        "sealed_protocol": {"weights_sha256": pipeline.digest(policy / "weights.pt")}}))
    adapter, alignment, exclusions = (tmp_path / name for name in ("adapter.json", "alignment.json", "exclusions.json"))
    for path in (adapter, alignment, exclusions):
        path.write_text("{}")
    output = tmp_path / "run"
    roots = [{"seed": i, "source_episode": 200 + i, "source_start_row": i * 201} for i in range(8)]
    monkeypatch.setattr(sys, "argv", ["pipeline", "--policy", str(policy), "--adapter-artifact", str(adapter),
        "--alignment-reference", str(alignment), "--disjoint-reference", str(exclusions), "--pilot-tracking", str(pilot),
        "--seed", "0", "--output-dir", str(output)])
    monkeypatch.setattr(pipeline, "bind_exclusions", lambda paths, seeds: ({100}, []))
    monkeypatch.setattr(pipeline.tracking_analysis, "load", lambda path: {})
    monkeypatch.setattr(pipeline.planner, "load_alignment", lambda path: {})
    monkeypatch.setattr(pipeline.shared, "receipt_source", lambda path: (tmp_path / "source.h5", "source_sha"))
    monkeypatch.setattr(pipeline, "planned_source_roots", lambda *args: roots)
    monkeypatch.setattr(pipeline.shared, "require_cuda_allocation", lambda device: {"fixture_cuda_contract": device})
    return output, roots


def test_all_four_stages_are_presealed_and_ordered(tmp_path, monkeypatch):
    output, roots = configure(tmp_path, monkeypatch)
    calls = []

    def run(command, check):
        protocol = json.loads((output / "pipeline.protocol.json").read_text())
        assert protocol["commands"][len(calls)] == command
        assert protocol["feedback_policy"]["iterations"] == 4
        assert check is True
        calls.append(command)
        if len(calls) <= 2:
            (output / ("planning.json" if len(calls) == 1 else "tracking.json")).write_text(json.dumps({"status": "completed", "gates": {"fixture": True}, "rows": roots}))

    monkeypatch.setattr(pipeline.subprocess, "run", run)
    pipeline.main()
    assert len(calls) == 4
    assert [c[c.index("-m") + 1] for c in calls] == [pipeline.planner.__name__, pipeline.tracker.__name__, pipeline.planning_analysis.__name__, pipeline.tracking_analysis.__name__]
    result = json.loads((output / "pipeline.json").read_text())
    assert result["status"] == "completed"
    assert all(v is True for v in result["gates"].values())


def test_child_failure_stops_without_claiming_completion(tmp_path, monkeypatch):
    output, _ = configure(tmp_path, monkeypatch)
    calls = []

    def run(command, check):
        calls.append(command)
        assert (output / "pipeline.protocol.json").exists()
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(pipeline.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.main()
    assert len(calls) == 1
    assert not (output / "pipeline.json").exists()
    assert not (output / "pipeline.stage_1.json").exists()


def test_metadata_collision_rejected_before_cuda_or_outputs(tmp_path, monkeypatch):
    output, _ = configure(tmp_path, monkeypatch)

    def collision(*args):
        raise ValueError("source-episode collision")

    monkeypatch.setattr(pipeline, "planned_source_roots", collision)
    monkeypatch.setattr(pipeline.shared, "require_cuda_allocation", lambda device: pytest.fail("CUDA must not execute after metadata rejection"))
    with pytest.raises(ValueError, match="collision"):
        pipeline.main()
    assert not output.exists()


def test_cuda_failure_never_falls_back_to_cpu(tmp_path, monkeypatch):
    output, _ = configure(tmp_path, monkeypatch)
    calls = []

    def cuda(device):
        calls.append(device)
        raise RuntimeError("CUDA unavailable")

    monkeypatch.setattr(pipeline.shared, "require_cuda_allocation", cuda)
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *args, **kwargs: pytest.fail("no CPU retry permitted"))
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        pipeline.main()
    assert calls == ["cuda"]
    assert (output / "pipeline.protocol.json").exists()
    assert not (output / "pipeline.json").exists()
