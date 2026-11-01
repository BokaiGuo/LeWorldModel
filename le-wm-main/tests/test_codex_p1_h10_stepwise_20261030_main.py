import math
import json

import pytest

from experiments.spectral_reachability import codex_p1_h10_stepwise_20261030_main as stepwise


def test_stepwise_metrics_distinguish_pusher_object_and_wrap_angle():
    start = [1., 2., 3., 4., 0., 0., 0.]
    goal = [4., 6., 6., 8., 0., 0., 0.]
    state = [4., 6., 3., 4., 2 * math.pi - .1, 0., 0.]
    result = stepwise.metrics(state, start, goal)
    assert result["pusher_displacement_xy"] == [3., 4.]
    assert result["pusher_goal_error"] == 0.
    assert result["object_displacement_norm"] == 0.
    assert result["object_position_error"] == 5.
    assert result["object_angle_error"] == pytest.approx(.1)


def test_stepwise_rejects_nonfinite_state():
    with pytest.raises(ValueError):
        stepwise.metrics([float("nan")] * 7, [0.] * 7, [0.] * 7)


def test_consume_checks_full_analyzer_then_binds_stepwise_rows(tmp_path, monkeypatch):
    calls = []
    anchors, shards, groups = [], [], []
    for i in range(24):
        role = stepwise.analysis.ROLES[i % 3]
        anchors.append({"role": role, "checkpoint": {"state": [0.] * 7},
                        "root_state": [0.] * 7, "goal": [0.] * 7})
        directory = tmp_path / f"anchor_{i:02d}"
        directory.mkdir()
        path = directory / "report.json"
        path.write_text(json.dumps({"rows": [{
            "index": 0, "status": "probe_completed",
            "responses": [{"step": 1, "state": [1., 0., 2., 0., 0., 0., 0.],
                           "terminated": False, "truncated": False}],
        }]}))
        shards.append({"report_sha256": stepwise.runner.runner.physical.sha(path)})
    for _ in range(8):
        groups.append({"probes": [{"roles": {role: {"status": "probe_completed"}
                                             for role in stepwise.analysis.ROLES}}]})
    report = tmp_path / "run_report.json"
    report.write_text(json.dumps({"shards": shards}))
    digest = stepwise.runner.runner.physical.sha(report)
    monkeypatch.setattr(stepwise.analysis, "consume", lambda *args: (
        calls.append("verified") or {"groups": groups, "status_counts": {"probe_completed": 24}}))
    monkeypatch.setattr(stepwise.runner, "verify_prepare", lambda *args: {"original": {"anchors": anchors}})
    result = stepwise.consume(str(tmp_path), digest, "/prepared", "a" * 64)
    assert calls == ["verified"]
    assert result["shards"][0]["rows"][0]["steps"][0]["metrics"]["pusher_displacement_xy"] == [1., 0.]
    assert result["shards"][0]["prefix_checkpoint_vs_root"]["object_position_error"] == 0.
    path = tmp_path / "anchor_00" / "report.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError):
        stepwise.consume(str(tmp_path), digest, "/prepared", "a" * 64)
