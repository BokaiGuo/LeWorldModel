import numpy as np

from experiments.gdd_wm.run_pointmaze_planning_shaped_control import (
    Objective,
    _feature,
    _feature_canonical_action_summary,
    _feature_ordered_action_sequence,
    _fit_counterfactual_endpoint,
    _fit_shared_future_scorer,
    _mean_predictor,
    _pair_indices,
    _per_root_decisions,
    _per_root_candidate_costs,
    _summary,
    endpoint_task_cost,
    _shared_future_action_feature,
    _shared_future_feature,
)


def _row(root: str, candidate: str, action: float) -> dict:
    return {
        "instance_id": root,
        "candidate_id": candidate,
        "source_history_model": [0.1, 0.2, 0.3, 0.4],
        "source_state_raw": [0.0, 0.0, 0.0, 0.0],
        "goal_raw": [1.0, 0.0],
        "action_sequence": [[action, 0.0], [action, 0.0]],
    }


def test_feature_uses_visual_history_goal_and_action_but_never_source_state():
    row = _row("root", "a", 0.5)
    row["source_state_raw"] = [999.0, 998.0, 997.0, 996.0]
    value = _feature(row)
    # action summary is mean/std/first/last (each XY) plus sequence length.
    assert value.shape == (4 + 2 + 9,)
    assert not np.isin([999.0, 998.0, 997.0, 996.0], value).any()


def test_ordered_action_feature_separates_sequences_collapsed_by_legacy_summary():
    first = np.zeros((20, 2), dtype=np.float64)
    first[1] = [1.0, 0.0]
    first[2] = [-1.0, 0.0]
    second = first.copy()
    second[[1, 2]] = second[[2, 1]]
    left, right = _row("root", "left", 0.0), _row("root", "right", 0.0)
    left["action_sequence"], right["action_sequence"] = first.tolist(), second.tolist()
    assert np.allclose(_feature(left), _feature(right))
    assert np.array_equal(_feature_canonical_action_summary(left), _feature_canonical_action_summary(right))
    assert not np.allclose(_feature_ordered_action_sequence(left), _feature_ordered_action_sequence(right))
    left["source_state_raw"] = [999.0, 998.0, 997.0, 996.0]
    assert not np.isin([999.0, 998.0, 997.0, 996.0], _feature_ordered_action_sequence(left)).any()


def test_pair_indices_point_from_lower_to_higher_cost_within_root_only():
    groups = {"a": [0, 1], "b": [2, 3]}
    preferred, rejected = _pair_indices(groups, np.asarray([2.0, 1.0, 3.0, 4.0]))
    assert list(zip(preferred.tolist(), rejected.tolist())) == [(1, 0), (2, 3)]


def test_summary_reports_ranking_and_effect_separately():
    rows = [_row("a", "left", 0.0), _row("a", "right", 1.0)]
    terminal = np.asarray([[0.1, 0.0, 0.0, 0.0], [0.9, 0.0, 0.0, 0.0]])
    cost = np.asarray([0.9, 0.1])
    result = _summary(rows, cost, np.asarray([0.8, 0.2]), np.zeros((2, 4)), terminal)
    assert result["pairwise_ranking_accuracy"] == 1.0
    assert result["mean_candidate_regret"] == 0.0
    assert result["effect_rmse"] > 0.0


def test_objective_contract_keeps_auxiliary_effect_and_ranking_terms_explicit():
    objective = Objective("planning_shaped", effect_weight=0.25, rank_weight=0.25)
    assert objective.effect_weight > 0.0
    assert objective.rank_weight > 0.0


def test_shared_seed_ensemble_averages_cost_and_effect_without_changing_input_contract():
    def first(values):
        return np.full(len(values), 2.0), np.full((len(values), 4), 4.0)

    def second(values):
        return np.full(len(values), 6.0), np.full((len(values), 4), 8.0)

    cost, effect = _mean_predictor([first, second])(np.zeros((3, 5)))
    assert np.all(cost == 4.0)
    assert np.all(effect == 6.0)


def test_per_root_decisions_preserve_paired_oracle_and_selected_regret():
    rows = [_row("a", "left", 0.0), _row("a", "right", 1.0)]
    decisions = _per_root_decisions(rows, np.asarray([0.9, 0.1]), np.asarray([0.8, 0.2]))
    assert decisions == [{
        "instance_id": "a", "candidate_count": 2,
        "oracle_candidate_id": "right", "selected_candidate_id": "right",
        "candidate_regret": 0.0, "zero_regret": True,
        "selected_predicted_cost": 0.2, "oracle_realized_cost": 0.1,
    }]


def test_per_root_candidate_cost_export_preserves_all_fixed_actions_and_costs():
    rows = [_row("a", "left", -0.5), _row("a", "right", 0.5)]
    rows[0]["action_sequence"] = [[-0.5, 0.0]] * 20
    rows[1]["action_sequence"] = [[0.5, 0.0]] * 20
    exported = _per_root_candidate_costs(rows, np.asarray([0.9, 0.1]), np.asarray([0.8, 0.2]))
    assert exported[0]["instance_id"] == "a"
    assert [row["candidate_id"] for row in exported[0]["candidates"]] == ["left", "right"]
    assert exported[0]["candidates"][1]["realized_cost"] == 0.1
    assert exported[0]["candidates"][1]["predicted_cost"] == 0.2


def test_counterfactual_endpoint_readout_uses_predicted_terminal_and_goal_only():
    endpoint = np.asarray([[0.0, 0.0, 9.0, 9.0], [1.0, 0.0, -3.0, 5.0]])
    goal = np.asarray([[1.0, 0.0], [1.0, 0.0]])
    assert endpoint_task_cost(endpoint, goal).tolist() == [1.0, 0.0]
    with np.testing.assert_raises_regex(ValueError, "endpoint"):
        endpoint_task_cost(np.zeros((2, 3)), goal)


def test_counterfactual_endpoint_predictor_derives_score_from_its_endpoint_on_cpu_unit_fixture():
    # CPU is intentional here: this is a small arithmetic/unit-contract test,
    # not a formal learned-world-model experiment.
    rows = [_row("a", "left", -0.5), _row("a", "right", 0.5), _row("b", "left", -0.25), _row("b", "right", 0.25)]
    features = np.stack([_feature(row) for row in rows])
    terminal = np.asarray([[0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0], [0.0, 0.2, 0.0, 0.0], [0.5, 0.2, 0.0, 0.0]])
    goal = np.asarray([row["goal_raw"] for row in rows], dtype=np.float64)
    _contract, predictor = _fit_counterfactual_endpoint(
        features, terminal, goal, {"a": [0, 1], "b": [2, 3]}, width=4, steps=2, device="cpu", seed=13,
    )
    score, endpoint = predictor(features, goal)
    assert endpoint.shape == (4, 4)
    assert np.isfinite(endpoint).all()
    assert np.allclose(score, endpoint_task_cost(endpoint, goal))


def test_shared_future_context_is_action_independent_but_scorer_receives_action():
    left, right = _row("root", "left", -0.5), _row("root", "right", 0.5)
    assert np.array_equal(_shared_future_feature(left), _shared_future_feature(right))
    assert not np.array_equal(_shared_future_action_feature(left), _shared_future_action_feature(right))


def test_shared_future_ordered_action_interface_separates_summary_collisions():
    first = np.zeros((20, 2), dtype=np.float64)
    first[1], first[2] = [1.0, 0.0], [-1.0, 0.0]
    second = first.copy()
    second[[1, 2]] = second[[2, 1]]
    left, right = _row("root", "left", 0.0), _row("root", "right", 0.0)
    left["action_sequence"], right["action_sequence"] = first.tolist(), second.tolist()
    assert np.array_equal(_shared_future_feature(left), _shared_future_feature(right))
    assert not np.array_equal(
        _shared_future_action_feature(left, "full_ordered_20x2"),
        _shared_future_action_feature(right, "full_ordered_20x2"),
    )


def test_shared_future_control_keeps_future_effect_equal_within_a_root_on_cpu_fixture():
    # CPU is intentional: this asserts the representation boundary, not a
    # learned-control result. Formal panels remain CUDA-only.
    rows = [_row("a", "left", -0.5), _row("a", "right", 0.5), _row("b", "left", -0.25), _row("b", "right", 0.25)]
    shared = np.stack([_shared_future_feature(row) for row in rows])
    action = np.stack([_shared_future_action_feature(row) for row in rows])
    effect = np.asarray([[0.0, 0.0, 0.0, 0.0], [0.4, 0.0, 0.0, 0.0], [0.0, 0.2, 0.0, 0.0], [0.5, 0.2, 0.0, 0.0]])
    cost = np.asarray([0.9, 0.1, 0.8, 0.2])
    contract, predictor = _fit_shared_future_scorer(
        shared, action, effect, cost, {"a": [0, 1], "b": [2, 3]}, width=4, steps=2, device="cpu", seed=17,
    )
    score, future_effect = predictor(shared, action)
    assert contract["shared_future_action_independent"] is True
    assert np.isfinite(score).all()
    assert np.allclose(future_effect[0], future_effect[1])
    assert np.allclose(future_effect[2], future_effect[3])
