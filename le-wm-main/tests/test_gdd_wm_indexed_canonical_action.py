import numpy as np

from experiments.gdd_wm.audit_indexed_canonical_action_geometry import (
    _bank_summary,
    _canonical_summary,
    _cyclic_prediction_shuffle,
    _features,
    _fit_indexed_bank,
)
from experiments.gdd_wm.aggregate_indexed_canonical_action_panel import (
    _paired_root_deltas,
)


def _row(source, goal, actions, candidate=0):
    return {
        "source_state_raw": [*source, 0.0, 0.0],
        "goal_raw": list(goal),
        "action_sequence": np.asarray(actions, dtype=float).tolist(),
        "candidate_id": candidate,
        "instance_id": f"root-{candidate}",
    }


def test_canonical_action_summary_is_rotation_invariant():
    canonical = np.asarray([[0.4, 0.2], [-0.1, 0.7]])
    source = np.asarray([0.0, 0.0])
    goal = np.asarray([1.0, 0.0])
    angle = 0.73
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    first = _row(source, goal, canonical)
    rotated = _row(rotation @ source, rotation @ goal, canonical @ rotation.T)
    assert np.allclose(_canonical_summary(first), _canonical_summary(rotated), atol=1.0e-10)


def test_train_derived_bank_rejects_inconsistent_candidate_controls():
    first = _row([0.0, 0.0], [1.0, 0.0], [[0.4, 0.2], [0.4, 0.2]], candidate=3)
    second = _row([0.0, 0.0], [1.0, 0.0], [[0.4, 0.4], [0.4, 0.4]], candidate=3)
    try:
        _fit_indexed_bank([first, second])
    except ValueError as error:
        assert "shared canonical action" in str(error)
    else:
        raise AssertionError("inconsistent candidate bank must fail closed")


def test_world_fixed_bank_summary_does_not_rotate_with_goal():
    actions = [[0.4, 0.2], [0.1, -0.3]]
    first = _row([0.0, 0.0], [1.0, 0.0], actions, candidate=2)
    second = _row([0.0, 0.0], [0.0, 1.0], actions, candidate=2)
    assert np.allclose(
        _bank_summary(first, "world_fixed"),
        _bank_summary(second, "world_fixed"),
    )
    assert not np.allclose(
        _bank_summary(first, "goal_relative"),
        _bank_summary(second, "goal_relative"),
    )


def test_shortcut_control_features_remove_declared_information():
    row = _row([0.0, 0.0], [1.0, 0.0], [[0.4, 0.2], [0.4, 0.2]], candidate=3)
    row["source_latent_model"] = [1.0, 2.0]
    row["_predicted"] = np.asarray([3.0, 6.0])
    bank = {3: _canonical_summary(row)}
    bank_only = _features(row, bank, family="bank_only")
    no_dynamics = _features(row, bank, family="no_dynamics_source_bank")
    full = _features(row, bank, family="full_source_delta_bank")
    shuffled = _features(
        row, bank, family="shuffled_prediction_full",
        predicted_override=np.asarray([30.0, 60.0]),
    )
    assert bank_only.shape == (12,)
    assert no_dynamics.shape == (14,)
    assert full.shape == shuffled.shape == (16,)
    assert np.allclose(no_dynamics[-12:], bank_only)
    assert not np.allclose(full, shuffled)


def test_prediction_shuffle_stays_within_root_and_has_no_fixed_points():
    rows = []
    for split in ("train", "outcome"):
        for root in ("a", "b"):
            for candidate in range(3):
                rows.append({
                    "split": split,
                    "instance_id": root,
                    "candidate_id": candidate,
                    "_predicted": np.asarray([100 * (split == "outcome") + 10 * (root == "b") + candidate]),
                })
    shuffled = _cyclic_prediction_shuffle(rows)
    for index, row in enumerate(rows):
        value = int(shuffled[index][0])
        source_prefix = 100 * (row["split"] == "outcome") + 10 * (row["instance_id"] == "b")
        assert value // 10 == source_prefix // 10
        assert value != int(row["_predicted"][0])


def test_paired_root_deltas_are_positive_when_full_is_better():
    full = {
        "a": {"regret": 0.1, "execution_consistency": 0.8},
        "b": {"regret": 0.2, "execution_consistency": 0.7},
    }
    control = {
        "a": {"regret": 0.3, "execution_consistency": 0.6},
        "b": {"regret": 0.4, "execution_consistency": 0.5},
    }
    deltas = _paired_root_deltas(full, control)
    assert np.allclose(deltas["regret_reduction"], [0.2, 0.2])
    assert np.allclose(deltas["consistency_gain"], [0.2, 0.2])
