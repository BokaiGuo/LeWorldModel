import numpy as np

from experiments.empirical_spectral_audit.atm_cross_domain import (
    action_information_gain,
    action_ranking_inversion,
    inverse_risk_matrix,
    transition_features,
)


def test_action_information_gain_is_risk_reduction_not_mutual_information():
    report = action_information_gain(2.0, 0.5)
    assert report["delta_action_information"] == 1.5
    assert "not Shannon" in report["semantics"]


def test_transition_features_exposes_effect_only_and_full_views():
    source = np.asarray([[1.0, 2.0]])
    successor = np.asarray([[2.0, 4.0]])
    assert transition_features(source, successor).shape == (1, 6)
    assert np.array_equal(transition_features(source, successor, effect_only=True), [[1.0, 2.0]])


def test_state_only_shortcut_is_reported_separately_from_effect_probe():
    source = np.asarray([[-1.0], [1.0], [-1.0], [1.0]])
    actions = source.copy()  # observational policy shortcut
    true_successor = source.copy()  # action has no effect
    predicted_successor = source.copy()
    report = inverse_risk_matrix(source, true_successor, predicted_successor, actions, alpha=1.0e-8)
    assert report["state_only_action_prior_risk"] < 1.0e-8
    assert report["effect_only_inverse_risk_matrix"][0][0] > 0.1
    assert report["protocol"]["effect_only_intervention_probe_reported"] is True


def test_ranking_inversion_is_finite_horizon_task_metric():
    report = action_ranking_inversion(
        real_cost=np.asarray([0.0, 2.0, 1.0, 0.0]),
        predicted_cost=np.asarray([2.0, 0.0, 1.0, 0.0]),
        group_id=np.asarray(["root0", "root0", "root0", "root1"]),
        minimum_margin=0.1,
    )
    assert report["inversion_count"] == 3
    assert report["comparable_pair_count"] == 3
    assert report["inversion_rate"] == 1.0
