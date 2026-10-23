from __future__ import annotations

import pytest

from experiments.gdd_wm import audit_pointmaze_candidate_design_frontier as frontier


def test_acceptance_checks_uses_v10_lower_tail_contract() -> None:
    passing = {
        "design_all_root_headroom": {
            "winner_entropy_normalized": 0.50,
            "best_second_gap_q10": 0.001,
            "best_second_gap_ge_0p003_fraction": 0.50,
            "candidate_cost_spread_q10": 0.05,
        }
    }
    assert all(frontier.acceptance_checks(passing).values())
    failing = {"design_all_root_headroom": {**passing["design_all_root_headroom"], "best_second_gap_q10": 0.000999}}
    assert frontier.acceptance_checks(failing)["top2_gap_lower_tail"] is False


def test_constructor_rejects_unregistered_design() -> None:
    with pytest.raises(ValueError, match="unsupported candidate design"):
        frontier._constructor("outcome_tuned_bank")


def test_pairwise_contrast_metrics_is_not_full_bank_argmin() -> None:
    def simulator(state, action):
        return [state[0] + action[:, 0].mean(), state[1] + action[:, 1].mean(), 0.0, 0.0]

    states = [[0.0, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0]]
    actions = [
        [[0.1, 0.0]] * 20, [[-0.1, 0.0]] * 20,
        [[0.0, 0.1]] * 20, [[0.0, -0.1]] * 20,
    ]
    report = frontier.pairwise_contrast_metrics(states, actions, [0.5, 0.2], simulator=simulator)
    assert report["pair_count"] == 2
    assert report["root_count"] == 2
    assert report["pooled_absolute_gap_q10"] > 0.0
