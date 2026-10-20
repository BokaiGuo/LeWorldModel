import json

import pytest

from experiments.summarize_cube_temporal_closed_loop import main


def _artifact(path, seeds, static, temporal, gate, fallback, feedback=None, *, approach_steps=0):
    rows = lambda values, counts=None: [
        {
            "seed": seed,
            "final_goal_distance_m": value,
            "support_fallback_count": 0 if counts is None else counts[index],
        }
        for index, (seed, value) in enumerate(zip(seeds, values))
    ]
    protocol = {
        "environment": "swm/OGBCube-v0",
        "checkpoint": "cube",
        "checkpoint_frozen": True,
        "fixed_prelude": "fixed",
        "approach_steps": approach_steps,
        "approach_gain": 1.0,
        "support_gate": "fixed",
        "support_threshold_m": 0.0005,
        "candidate_bank": [[0.0], [1.0]],
        "runtime_input": "rgb",
        "runtime_policy_state_access": False,
    }
    payload = {
        "schema": "cube_temporal_closed_loop.v1",
        "protocol": protocol,
        "paired": {"seed_order": seeds},
        "static": {"episodes": rows(static)},
        "temporal": {"episodes": rows(temporal)},
        "temporal_support_gate": {"episodes": rows(gate, fallback)},
    }
    if feedback is not None:
        payload["temporal_pixel_feedback"] = {"episodes": rows(feedback)}
    path.write_text(json.dumps(payload))


def test_closed_loop_aggregate_reports_paired_ci_and_support_strata(tmp_path, monkeypatch):
    first = tmp_path / "first.json"
    _artifact(first, [1, 2, 3], [1.0, 1.0, 1.0], [0.5, 1.0, 1.5], [0.5, 1.0, 1.5], [1, 0, 0])
    output = tmp_path / "aggregate.json"
    monkeypatch.setattr("sys.argv", ["summarize", "--input", str(first), "--output", str(output)])
    main()
    report = json.loads(output.read_text())
    assert report["schema"] == "cube_temporal_closed_loop.aggregate.v1"
    pooled = report["pooled"]
    assert pooled["temporal_minus_static"]["wins_improvement"] == 1
    assert pooled["temporal_minus_static"]["losses_deterioration"] == 1
    assert pooled["temporal_minus_static"]["ties"] == 1
    assert pooled["support_triggered_episodes"] == 1
    assert pooled["support_triggered_gate_minus_static"]["n"] == 1


def test_closed_loop_aggregate_rejects_overlapping_panels(tmp_path, monkeypatch):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    args = ([1], [1.0], [0.5], [0.5], [1])
    _artifact(first, *args)
    _artifact(second, *args)
    monkeypatch.setattr(
        "sys.argv", ["summarize", "--input", str(first), "--input", str(second), "--output", str(tmp_path / "x.json")]
    )
    with pytest.raises(ValueError, match="overlap"):
        main()


def test_closed_loop_aggregate_reports_optional_pixel_feedback(tmp_path, monkeypatch):
    artifact = tmp_path / "feedback.json"
    _artifact(
        artifact,
        [1, 2, 3],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [0, 0, 0],
        feedback=[0.5, 1.0, 0.75],
    )
    output = tmp_path / "feedback_aggregate.json"
    monkeypatch.setattr("sys.argv", ["summarize", "--input", str(artifact), "--output", str(output)])
    main()
    report = json.loads(output.read_text())
    assert report["pooled"]["pixel_feedback_minus_static"]["wins_improvement"] == 2
    assert report["pooled"]["pixel_feedback_minus_static"]["losses_deterioration"] == 0


def test_closed_loop_aggregate_rejects_mismatched_approach_protocol(tmp_path, monkeypatch):
    first = tmp_path / "no_approach.json"
    second = tmp_path / "approach.json"
    args = ([1], [1.0], [0.5], [0.5], [0])
    _artifact(first, *args, approach_steps=0)
    _artifact(second, *args, approach_steps=10)
    monkeypatch.setattr(
        "sys.argv",
        ["summarize", "--input", str(first), "--input", str(second), "--output", str(tmp_path / "x.json")],
    )
    with pytest.raises(ValueError, match="protocol mismatch"):
        main()
