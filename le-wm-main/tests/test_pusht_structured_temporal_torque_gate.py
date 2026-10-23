import numpy as np
from pathlib import Path

from experiments.run_pusht_balanced_temporal_torque_confirmation import RootExample
from experiments.run_pusht_structured_temporal_torque_gate import (
    confirmation_rule,
    select_confidence_gate,
    structured_temporal_feature,
)


def _row() -> RootExample:
    frames = np.zeros((5, 12, 14, 3), dtype=np.uint8)
    for index in range(len(frames)):
        frames[index, index : index + 3, index : index + 3, :] = 30 * (index + 1)
    actions = np.asarray(((0.3, 0.0), (0.0, 0.3), (-0.3, 0.0), (0.0, -0.3)), dtype=np.float32)
    return RootExample(1, "digest", frames, actions, 1, 0.01, 0.02, 0.01)


def test_structured_chart_is_temporal_and_low_dimension():
    row = _row()
    feature = structured_temporal_feature(row.rgb_frames, row.history_actions, grid_size=4)
    # final RGB + four RGB differences = five 3x4x4 maps, plus four 2-D actions
    assert feature.shape == (5 * 3 * 4 * 4 + 4 * 2,)
    altered = row.rgb_frames.copy()
    altered[0] = 255
    assert not np.array_equal(
        feature,
        structured_temporal_feature(altered, row.history_actions, grid_size=4),
    )


def test_gate_uses_only_calibration_quality_and_fails_closed_without_support():
    probability = np.asarray((0.01, 0.02, 0.98, 0.99, 0.49, 0.51, 0.48, 0.52))
    target = np.asarray((0, 0, 1, 1, 0, 1, 0, 1))
    gate = select_confidence_gate(
        probability, target, minimum_calibration_ba=0.90, minimum_coverage=0.25, minimum_per_class=2
    )
    assert gate["authorized"]
    assert gate["threshold"] == 0.0
    rejected = select_confidence_gate(
        probability, target, minimum_calibration_ba=1.01, minimum_coverage=0.25, minimum_per_class=2
    )
    assert not rejected["authorized"]
    assert rejected["threshold"] is None


def test_confirmation_requires_gate_coverage_and_held_quality():
    gate = {"authorized": True}
    outcome = {
        "executed": True,
        "coverage": 0.5,
        "balanced_accuracy": {"estimate": 0.8, "ci95_low": 0.65},
    }
    assert confirmation_rule(gate, outcome, minimum_outcome_ba=0.70, minimum_outcome_ci_lower=0.60, minimum_outcome_coverage=0.25)["status"] == "confirmed_selective_offline_transfer"
    outcome["balanced_accuracy"]["ci95_low"] = 0.59
    assert confirmation_rule(gate, outcome, minimum_outcome_ba=0.70, minimum_outcome_ci_lower=0.60, minimum_outcome_coverage=0.25)["status"] == "not_confirmed"


def test_gpu_launcher_freezes_fresh_protocol_and_refuses_output_reuse():
    launcher = Path(__file__).resolve().parents[1] / "experiments" / "run_pusht_structured_temporal_torque_gate_cuda.sh"
    text = launcher.read_text(encoding="utf-8")
    assert "run_with_torch_cudnn.sh .venv/bin/python -m" in text
    assert "run_pusht_structured_temporal_torque_gate" in text
    assert "Refusing to reuse immutable protocol output" in text
    for frozen in ("--train-per-class 128", "--calibration-per-class 64", "--test-per-class 128", "--grid-size 4", "--pca-rank 24"):
        assert frozen in text
