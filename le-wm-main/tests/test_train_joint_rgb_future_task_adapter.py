import numpy as np
import torch

from experiments.decision_benchmark.train_joint_rgb_future_task_adapter import (
    AsymmetricFutureTaskMLP,
    JointFutureTaskMLP,
    _action_sequence_in_frame,
    _reference_index_by_root,
    _root_ranking,
    _split_report,
)


def test_joint_model_has_separate_future_and_task_heads():
    model = JointFutureTaskMLP(input_dim=13, rgb_dim=8, task_dim=2)
    rgb, task, cost = model(torch.zeros((3, 13), dtype=torch.float32))
    assert tuple(rgb.shape) == (3, 8)
    assert tuple(task.shape) == (3, 2)
    assert tuple(cost.shape) == (3, 1)


def test_asymmetric_model_has_future_action_interface():
    model = AsymmetricFutureTaskMLP(context_dim=5, action_dim=8, rgb_dim=3, task_dim=2)
    rgb, task, cost = model(
        torch.zeros((4, 5), dtype=torch.float32),
        torch.zeros((4, 8), dtype=torch.float32),
    )
    assert tuple(rgb.shape) == (4, 3)
    assert tuple(task.shape) == (4, 2)
    assert tuple(cost.shape) == (4, 1)


def test_asymmetric_model_can_emit_future_sequence():
    model = AsymmetricFutureTaskMLP(context_dim=5, action_dim=8, rgb_dim=3, future_steps=4)
    rgb, task, cost = model(
        torch.zeros((2, 5), dtype=torch.float32),
        torch.zeros((2, 8), dtype=torch.float32),
    )
    assert tuple(rgb.shape) == (2, 4, 3)
    assert tuple(task.shape) == (2, 2)
    assert tuple(cost.shape) == (2, 1)


def test_asymmetric_model_can_use_direct_future_sequence_readout():
    model = AsymmetricFutureTaskMLP(
        context_dim=5, action_dim=8, rgb_dim=3, task_dim=2, future_steps=4,
        direct_future_readout=True,
    )
    rgb, task, cost = model(
        torch.zeros((2, 5), dtype=torch.float32),
        torch.zeros((2, 8), dtype=torch.float32),
    )
    assert tuple(rgb.shape) == (2, 4, 3)
    assert tuple(task.shape) == (2, 2)
    assert tuple(cost.shape) == (2, 1)


def test_asymmetric_relative_task_decomposes_context_and_action_effect():
    model = AsymmetricFutureTaskMLP(
        context_dim=5, action_dim=8, rgb_dim=3, task_dim=2, relative_task=True,
    )
    rgb, task, cost = model(
        torch.zeros((4, 5), dtype=torch.float32),
        torch.zeros((4, 8), dtype=torch.float32),
    )
    assert tuple(rgb.shape) == (4, 3)
    assert tuple(task.shape) == (4, 2)
    assert tuple(cost.shape) == (4, 1)


def test_reference_factorized_effect_requires_and_accepts_reference_action():
    model = AsymmetricFutureTaskMLP(
        context_dim=5, action_dim=8, rgb_dim=3, task_dim=2,
        relative_task=True, reference_factorized=True,
    )
    context = torch.zeros((4, 5), dtype=torch.float32)
    action = torch.zeros((4, 8), dtype=torch.float32)
    reference = torch.zeros((4, 8), dtype=torch.float32)
    rgb, task, cost = model(context, action, reference)
    assert tuple(rgb.shape) == (4, 3)
    assert tuple(task.shape) == (4, 2)
    assert tuple(cost.shape) == (4, 1)


def test_reference_factorized_effect_rejects_missing_reference_action():
    model = AsymmetricFutureTaskMLP(
        context_dim=5, action_dim=8, rgb_dim=3, task_dim=2,
        relative_task=True, reference_factorized=True,
    )
    try:
        model(torch.zeros((2, 5)), torch.zeros((2, 8)))
    except ValueError as error:
        assert "reference_action" in str(error)
    else:
        raise AssertionError("reference-factorized model must require reference_action")


def test_goal_relative_action_frame_aligns_root_rotations():
    canonical = np.asarray([[0.6, 0.2], [0.0, -0.4]], dtype=np.float64)
    first = {
        "action_sequence": canonical.tolist(),
        "source_state_raw": [0.0, 0.0, 0.0, 0.0],
        "goal_raw": [1.0, 0.0],
    }
    # The second root has a 90-degree rotated raw control frame.
    second = {
        "action_sequence": np.stack([-canonical[:, 1], canonical[:, 0]], axis=1).tolist(),
        "source_state_raw": [0.0, 0.0, 0.0, 0.0],
        "goal_raw": [0.0, 1.0],
    }
    np.testing.assert_allclose(
        _action_sequence_in_frame(first, "goal_relative"),
        _action_sequence_in_frame(second, "goal_relative"),
    )


def test_hybrid_action_frame_preserves_raw_and_goal_relative_channels():
    record = {
        "action_sequence": [[0.6, 0.2], [0.0, -0.4]],
        "source_state_raw": [0.0, 0.0, 0.0, 0.0],
        "goal_raw": [0.0, 1.0],
    }
    hybrid = _action_sequence_in_frame(record, "hybrid")
    assert hybrid.shape == (2, 4)
    np.testing.assert_allclose(hybrid[:, :2], np.asarray(record["action_sequence"]))
    np.testing.assert_allclose(
        hybrid[:, 2:], _action_sequence_in_frame(record, "goal_relative"),
    )


def test_explicit_reference_candidate_is_not_replaced_by_minimum_energy():
    records = []
    for candidate_id, energy in ((0, 0.0), (6, 0.4)):
        records.append({
            "instance_id": "root0",
            "candidate_id": candidate_id,
            "action_sequence": [[energy, 0.0], [energy, 0.0]],
        })
    selected = _reference_index_by_root(records, reference_candidate_id=6)
    assert selected["root0"] == 1


def test_root_ranking_uses_predicted_terminal_but_scores_real_terminal():
    roots = ["r0", "r0", "r1", "r1"]
    goals = np.zeros((4, 2), dtype=np.float64)
    actual = np.asarray([[0.1, 0.0], [0.4, 0.0], [0.2, 0.0], [0.3, 0.0]])
    predicted = np.asarray([[0.05, 0.0], [0.5, 0.0], [0.25, 0.0], [0.1, 0.0]])
    report = _root_ranking(roots, predicted, actual, goals, success_threshold=0.25)
    assert report["roots"] == 2
    assert report["selected_success_rate"] == 0.5
    assert np.isfinite(report["paired_regret_gain_mean"])


def test_non_cost_objective_does_not_use_cost_head_for_ranking():
    data = {
        "task_target": np.zeros((2, 2), dtype=np.float64),
        "rgb_target": np.zeros((2, 1), dtype=np.float64),
        "source_xy": np.zeros((2, 2), dtype=np.float64),
        "goals": np.zeros((2, 2), dtype=np.float64),
        "actual_terminal_xy": np.asarray([[0.05, 0.0], [0.4, 0.0]], dtype=np.float64),
        "roots": ["r0", "r0"],
    }
    predictions = {
        "task_prediction": np.asarray([[0.05, 0.0], [0.4, 0.0]], dtype=np.float64),
        "rgb_prediction": np.zeros((2, 1), dtype=np.float64),
        # Deliberately points to the worse action; this must be ignored unless
        # the objective explicitly enables the direct cost head.
        "cost_prediction": np.asarray([0.9, 0.1], dtype=np.float64),
    }
    report = _split_report(
        data, predictions, np.asarray([True, True]), success_threshold=0.25,
        allow_cost_head=False,
    )
    assert report["ranking"]["direct_cost_head_used"] is False
    assert report["ranking"]["selected_regret_mean"] == 0.0
