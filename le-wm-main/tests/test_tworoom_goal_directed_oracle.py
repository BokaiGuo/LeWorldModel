import numpy as np

from experiments.audit_tworoom_goal_directed_oracle import _candidate_bank


def test_goal_directed_oracle_bank_is_root_local_and_well_formed() -> None:
    candidates, labels = _candidate_bank(
        np.array([0.1, -0.2]),
        np.array([0.8, 0.6]),
        horizon=5,
        action_block=5,
        magnitudes=(0.25, 0.5, 0.75, 1.0),
    )
    assert candidates.shape == (289, 25, 2)
    assert len(labels) == candidates.shape[0]
    assert np.isfinite(candidates).all()
    assert np.max(np.abs(candidates)) <= 1.0
    assert {label["family"] for label in labels} == {
        "zero", "constant", "two_stage_goal_directed",
    }


def test_goal_directed_oracle_rejects_degenerate_goal() -> None:
    with np.testing.assert_raises(ValueError):
        _candidate_bank(
            np.zeros(2), np.zeros(2), horizon=5, action_block=5,
            magnitudes=(0.5,),
        )
