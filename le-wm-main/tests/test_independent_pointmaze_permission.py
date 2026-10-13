import numpy as np

from experiments.empirical_spectral_audit.build_independent_pointmaze_planning_gain import (
    _select_candidate_indices,
)


def test_rejected_permission_forces_baseline_candidate():
    predicted = np.asarray([[0.0, 10.0, 20.0], [0.0, 5.0, 6.0]])
    selected = _select_candidate_indices(
        predicted,
        {"planning_permission": "rejected_fail_closed", "candidate_horizons_allowed": [1, 2, 4]},
        baseline_horizon=1,
    )
    assert selected.tolist() == [0, 0]


def test_accepted_permission_masks_disallowed_horizon():
    predicted = np.asarray([[0.0, 10.0, 20.0], [0.0, 5.0, 6.0]])
    selected = _select_candidate_indices(
        predicted,
        {"planning_permission": "accepted", "candidate_horizons_allowed": [1, 2]},
        baseline_horizon=1,
    )
    assert selected.tolist() == [1, 1]
