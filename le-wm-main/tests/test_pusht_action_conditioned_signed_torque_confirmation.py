from pathlib import Path

import numpy as np

from experiments.run_pusht_action_conditioned_signed_torque_confirmation import (
    ActionConditionedExample,
    body_frame_mechanics_features,
    candidate_pair_features,
    contact_action_mechanics_features,
    geometry_history_features,
)


def _row(plus=(0.2, 0.5), minus=(0.2, -0.5)) -> ActionConditionedExample:
    frames = np.zeros((5, 12, 14, 3), dtype=np.uint8)
    frames[:, 2:6, 2:6, 2] = 255
    frames[:, 6:10, 7:12, :] = np.asarray((110, 130, 150), dtype=np.uint8)
    return ActionConditionedExample(
        root_id=1,
        root_digest="root",
        rgb_frames=frames,
        history_actions=np.asarray(((0.3, 0.0), (0.0, 0.3), (-0.3, 0.0), (0.0, -0.3)), dtype=np.float32),
        plus_action=np.asarray(plus, dtype=np.float32),
        minus_action=np.asarray(minus, dtype=np.float32),
        preference=1,
        preference_margin=0.01,
        zero_corrected_plus_response=0.02,
        zero_corrected_minus_response=0.01,
    )


def test_candidate_pair_is_explicit_runtime_feature() -> None:
    first = _row()
    second = _row(plus=(0.5, 0.2), minus=(-0.5, 0.2))
    values = candidate_pair_features([first, second])
    assert values.shape == (2, 4)
    assert not np.array_equal(values[0], values[1])


def test_geometry_history_is_rgb_and_history_derived_only() -> None:
    row = _row()
    feature = geometry_history_features([row], device="cpu")
    assert feature.shape[0] == 1
    assert feature.shape[1] > row.history_actions.size
    changed = _row()
    changed.rgb_frames[0] = 0
    assert not np.array_equal(feature, geometry_history_features([changed], device="cpu"))


def test_mechanics_chart_contains_rgb_lever_arm_candidate_cross_terms() -> None:
    first = _row()
    second = _row(plus=(0.5, 0.2), minus=(-0.5, 0.2))
    geometry = geometry_history_features([first, second], device="cpu")
    mechanics = contact_action_mechanics_features(geometry, [first, second])
    assert mechanics.shape == (2, 8)
    assert not np.array_equal(mechanics[0], mechanics[1])


def test_body_frame_chart_exposes_oriented_rgb_geometry_action_interactions() -> None:
    first = _row()
    second = _row(plus=(0.5, 0.2), minus=(-0.5, 0.2))
    geometry = geometry_history_features([first, second], device="cpu")
    values = body_frame_mechanics_features(geometry, [first, second])
    assert values.shape == (2, 8)
    assert not np.array_equal(values[0], values[1])


def test_cuda_launcher_freezes_the_action_conditioned_contract() -> None:
    launcher = Path(__file__).resolve().parents[1] / "experiments" / "run_pusht_action_conditioned_signed_torque_confirmation_cuda.sh"
    text = launcher.read_text(encoding="utf-8")
    assert "run_with_torch_cudnn.sh .venv/bin/python -m experiments.gpu_preflight" in text
    assert "run_pusht_action_conditioned_signed_torque_confirmation" in text
    assert "Refusing to reuse immutable protocol output" in text
    assert "--train-per-class 128 --calibration-per-class 64 --test-per-class 128" in text
    assert "--confirmation-representation geometry_plus_rigid_body_mechanics" in text
