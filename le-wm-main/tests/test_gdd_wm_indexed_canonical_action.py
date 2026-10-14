import numpy as np

from experiments.gdd_wm.audit_indexed_canonical_action_geometry import (
    _canonical_summary,
    _fit_indexed_bank,
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
