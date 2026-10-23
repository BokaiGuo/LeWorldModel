import numpy as np
import pytest
import json

from experiments.gdd_wm.run_pointmaze_planning_shaped_repairability import (
    _calibration_partition,
    _calibrate_policy,
    _deployment_summary,
    _evaluate,
    _fit_ridge_advantage,
    _fit_bootstrap_lower_advantage,
    _fit_selected_ridge_advantage,
    _fit_logistic,
    _feature_for_action_representation,
    _opportunity_compact_features,
    _post_execution_labels,
    _root_fold,
    _root_features,
    _summary,
    _upper_bootstrap_delta,
    _validate_shared_candidate_support,
    _validate_root_partition,
)
from experiments.gdd_wm.run_pointmaze_planning_shaped_control import _feature


def _row(candidate: str, action: float) -> dict:
    return {
        "candidate_id": candidate,
        "source_history_model": [0.1, 0.2, 0.3, 0.4],
        "source_latent_model": [0.3, 0.4],
        "source_state_raw": [999.0, 998.0, 997.0, 996.0],
        "goal_raw": [1.0, 0.0],
        "action_sequence": [[action, 0.0]] * 20,
    }


def test_root_features_exclude_raw_simulator_state() -> None:
    group = [_row("left", -0.5), _row("right", 0.5)]
    features = _root_features(
        group, np.asarray([0.1, 0.3]), np.asarray([0.2, 0.1]),
        np.asarray([[0.1, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0]]),
        np.asarray([[0.3, 0.0, 0.0, 0.0], [0.4, 0.0, 0.0, 0.0]]),
    )
    assert features.shape == (15,)
    assert not np.isin([999.0, 998.0, 997.0, 996.0], features).any()


def test_root_features_require_paired_effect_predictions() -> None:
    group = [_row("left", -0.5), _row("right", 0.5)]
    with pytest.raises(ValueError, match="supplied together"):
        _root_features(group, np.asarray([0.1, 0.3]), np.asarray([0.2, 0.1]), np.zeros((2, 4)))


def test_constant_crossfit_labels_yield_legal_constant_router() -> None:
    router = _fit_logistic(np.zeros((3, 2)), np.ones(3))
    assert np.allclose(router(np.ones((4, 2))), 1.0)


def test_advantage_router_preserves_signed_crossfit_utility() -> None:
    router = _fit_ridge_advantage(
        np.asarray([[0.0], [1.0], [2.0], [3.0]]),
        np.asarray([-0.3, -0.1, 0.2, 0.6]),
        l2=0.0,
    )
    score = router(np.asarray([[0.0], [3.0]]))
    assert score[0] < 0.0 < score[1]


def test_advantage_router_selects_regularization_on_root_blocked_train_labels_only() -> None:
    roots = [f"train-root-{index}" for index in range(24)]
    features = np.asarray([[float(index), float(index % 3)] for index in range(24)])
    advantage = np.asarray([float(index - 12) / 12.0 for index in range(24)])
    router, audit = _fit_selected_ridge_advantage(roots, features, advantage, l2_grid=(0.25, 4.0, 64.0))
    assert audit["selection_protocol"] == "train_root_only_four_fold_blocked_advantage_mse"
    assert audit["root_count"] == len(roots)
    assert audit["feature_dimension"] == features.shape[1]
    assert [row["l2"] for row in audit["candidates"]] == [0.25, 4.0, 64.0]
    assert audit["selected"]["l2"] in {0.25, 4.0, 64.0}
    score = router(np.asarray([[0.0, 0.0], [23.0, 2.0]]))
    assert score[0] < score[1]


def test_advantage_router_rejects_duplicate_root_identities() -> None:
    with pytest.raises(ValueError, match="unique"):
        _fit_selected_ridge_advantage(
            ["same"] * 8, np.zeros((8, 2)), np.zeros(8), l2_grid=(1.0,),
        )


def test_bootstrap_lower_advantage_router_is_train_only_and_conservative() -> None:
    roots = [f"bootstrap-root-{index}" for index in range(16)]
    features = np.asarray([[float(index), float(index % 2)] for index in range(16)])
    advantage = np.asarray([float(index - 8) / 8.0 for index in range(16)])
    router, audit = _fit_bootstrap_lower_advantage(
        roots, features, advantage, seed=17, member_count=16, lower_quantile=0.10,
    )
    score = router(np.asarray([[0.0, 0.0], [15.0, 1.0]]))
    assert score.shape == (2,)
    assert score[0] < score[1]
    assert audit["uncertainty_transform"] == "train_root_bootstrap_lower_advantage_quantile"
    assert audit["bootstrap_member_count"] == 16


def test_ordered_action_contract_breaks_legacy_summary_collision() -> None:
    def collision_row(candidate: str, interior: list[float]) -> dict:
        return {
            **_row(candidate, 0.0),
            "action_sequence": [[0.0, 0.0]] + [[value, 0.0] for value in interior] + [[0.0, 0.0]],
        }

    left = collision_row("left", [1.0] * 9 + [-1.0] * 9)
    right = collision_row("right", [-1.0] * 9 + [1.0] * 9)
    assert np.allclose(_feature(left), _feature(right))
    ordered = _feature_for_action_representation("full_ordered_20x2")
    assert not np.allclose(ordered(left), ordered(right))
    assert _feature_for_action_representation("summary_mean_std_first_last") is _feature
    with pytest.raises(ValueError, match="action representation"):
        _feature_for_action_representation("unknown")


def test_ordered_router_features_retain_the_selected_action_sequences() -> None:
    def collision_row(candidate: str, interior: list[float]) -> dict:
        return {
            **_row(candidate, 0.0),
            "action_sequence": [[0.0, 0.0]] + [[value, 0.0] for value in interior] + [[0.0, 0.0]],
        }

    left = collision_row("left", [1.0] * 9 + [-1.0] * 9)
    right = collision_row("right", [-1.0] * 9 + [1.0] * 9)
    legacy = _root_features(
        [left, right], np.asarray([0.1, 0.3]), np.asarray([0.3, 0.1]),
        action_representation="summary_mean_std_first_last",
    )
    ordered = _root_features(
        [left, right], np.asarray([0.1, 0.3]), np.asarray([0.3, 0.1]),
        action_representation="full_ordered_20x2",
    )
    assert legacy.shape == (11,)
    assert ordered.shape == (91,)
    assert np.allclose(ordered[-80:-40], np.asarray(left["action_sequence"]).reshape(-1))
    assert np.allclose(ordered[-40:], np.asarray(right["action_sequence"]).reshape(-1))


def test_v9_compact_opportunity_features_match_declared_capacity_and_retain_signed_action_difference() -> None:
    def ordered_row(candidate: str, interior: list[float]) -> dict:
        return {
            **_row(candidate, 0.0),
            "action_sequence": [[0.0, 0.0]] + [[value, 0.0] for value in interior] + [[0.0, 0.0]],
        }

    left = ordered_row("left", [1.0] * 9 + [-1.0] * 9)
    right = ordered_row("right", [-1.0] * 9 + [1.0] * 9)
    values = _opportunity_compact_features(
        [left, right], np.asarray([0.1, 0.4]), np.asarray([0.3, 0.2]),
    )
    assert values.shape == (15,)
    # The last coordinate measures the early-vs-late signed difference, so
    # V9's compact router does not silently reduce an order reversal to a norm.
    assert values[-1] > 0.0


def test_ordered_scorers_receive_the_same_action_schema_at_evaluation() -> None:
    def ordered_row(candidate: str, interior: list[float]) -> dict:
        return {
            **_row(candidate, 0.0),
            "action_sequence": [[0.0, 0.0]] + [[value, 0.0] for value in interior] + [[0.0, 0.0]],
        }

    group = [
        ordered_row("left", [1.0] * 9 + [-1.0] * 9),
        ordered_row("right", [-1.0] * 9 + [1.0] * 9),
    ]
    ordered_feature = _feature_for_action_representation("full_ordered_20x2")
    expected_dimension = len(ordered_feature(group[0]))

    def predictor(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert features.shape == (2, expected_dimension)
        return np.asarray([0.1, 0.2]), np.zeros((2, 4))

    result = _evaluate(
        {"root": group}, {"root": np.asarray([0.0, 1.0])}, predictor, predictor,
        lambda values: np.zeros(len(values)), threshold=0.0,
        router_kind="advantage_ridge", action_representation="full_ordered_20x2",
    )
    assert result["root"]["cost_candidate_id"] == "left"


def test_opportunity_gate_never_releases_when_scorers_choose_the_same_candidate() -> None:
    group = [_row("left", -0.5), _row("right", 0.5)]

    def predictor(_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray([0.1, 0.2]), np.zeros((2, 4))

    def router(values: np.ndarray) -> np.ndarray:
        assert values.shape == (1, 15)
        return np.full(len(values), 100.0)

    result = _evaluate(
        {"root": group}, {"root": np.asarray([0.0, 1.0])}, predictor, predictor,
        router, threshold=0.0,
        router_kind="advantage_ridge", action_representation="summary_mean_std_first_last",
        opportunity_gated=True,
    )["root"]
    assert result["candidate_disagreement"] is False
    assert result["routing_eligible"] is False
    assert result["released_planning"] is False
    assert result["routed_candidate_id"] == result["cost_candidate_id"]


def test_opportunity_calibration_uses_only_declared_disagreement_roots() -> None:
    rows = {
        f"root-{index}": {
            "router_score_pre_outcome": 0.0,
            "planning_beats_cost": False,
            "routing_eligible": False,
            "cost_regret": 0.0,
            "planning_regret": 0.0,
            "routed_regret": 0.0,
        }
        for index in range(160)
    }
    selector, certification = _calibration_partition(rows)
    for partition in (selector, certification):
        for index, root in enumerate(list(partition)[:8]):
            rows[root].update({
                "router_score_pre_outcome": 1.0 if index < 4 else 0.0,
                "planning_beats_cost": index < 4,
                "routing_eligible": True,
                "cost_regret": 1.0,
                "planning_regret": 0.0 if index < 4 else 2.0,
            })
    policy = _calibrate_policy(
        rows, seed=29, opportunity_gated=True,
        minimum_opportunity_roots_per_role=8,
    )
    assert policy["routing_authorized"] is True
    assert policy["fixed_incumbent"] == "cost"
    assert policy["selector_opportunity_root_count"] == 8
    assert policy["certification_opportunity_root_count"] == 8
    assert policy["selector_summary"]["planning_release_coverage"] < 0.10
    assert policy["opportunity_gated"] is True


def test_opportunity_calibration_rejects_an_underpowered_disagreement_stratum() -> None:
    rows = {
        f"root-{index}": {
            "router_score_pre_outcome": 1.0,
            "planning_beats_cost": True,
            "routing_eligible": False,
            "cost_regret": 1.0,
            "planning_regret": 0.0,
            "routed_regret": 0.0,
        }
        for index in range(64)
    }
    selector, certification = _calibration_partition(rows)
    next(iter(selector.values()))["routing_eligible"] = True
    next(iter(certification.values()))["routing_eligible"] = True
    with pytest.raises(ValueError, match="minimum number"):
        _calibrate_policy(
            rows, seed=31, opportunity_gated=True,
            minimum_opportunity_roots_per_role=2,
        )


def test_summary_distinguishes_zero_coverage_from_repair_precision() -> None:
    rows = {
        "root": {
            "planning_beats_cost": True,
            "released_planning": False,
            "cost_regret": 0.2,
            "planning_regret": 0.0,
            "routed_regret": 0.2,
        }
    }
    value = _summary(rows)
    assert value["planning_release_coverage"] == 0.0
    assert value["release_precision"] is None
    assert value["release_recall"] == 0.0


def test_calibration_certificate_falls_back_when_routing_has_no_strict_holdout_gain() -> None:
    rows = {
        f"root-{index}": {
            "repair_probability_pre_outcome": 0.9,
            "released_planning": True,
            "planning_beats_cost": False,
            "cost_regret": 1.0,
            "planning_regret": 1.0,
            "routed_regret": 1.0,
        }
        for index in range(24)
    }
    policy = _calibrate_policy(rows, seed=7)
    assert policy["fixed_incumbent"] == "cost"
    assert policy["routing_authorized"] is False
    assert policy["deployment_policy"] == "cost"
    assert policy["selector_root_count"] + policy["certification_root_count"] == len(rows)


def test_paired_upper_bound_recognizes_clear_improvement() -> None:
    upper = _upper_bootstrap_delta(np.zeros(12), np.ones(12), seed=17)
    assert upper < 0.0


def test_calibration_certificate_authorizes_a_nontrivial_routing_gain() -> None:
    rows = {
        f"root-{index}": {
            "repair_probability_pre_outcome": 0.1,
            "released_planning": False,
            "planning_beats_cost": False,
            "cost_regret": 0.0,
            "planning_regret": 1.0,
            "routed_regret": 0.0,
        }
        for index in range(80)
    }
    selector, certification = _calibration_partition(rows)
    for subset in (selector, certification):
        for index, row in enumerate(subset.values()):
            if index < 4:
                row.update({
                    "repair_probability_pre_outcome": 0.9,
                    "released_planning": True,
                    "planning_beats_cost": True,
                    "cost_regret": 1.0,
                    "planning_regret": 0.0,
                })
    rows = {**selector, **certification}
    policy = _calibrate_policy(rows, seed=11)
    assert policy["fixed_incumbent"] == "cost"
    assert policy["routing_authorized"] is True
    assert policy["deployment_policy"] == "routed"
    assert policy["selector_summary"]["planning_release_coverage"] == 4 / policy["selector_root_count"]


def test_fixed_deployment_summary_reports_fixed_arm_coverage_not_stale_router_coverage() -> None:
    rows = {
        "root": {
            "repair_probability_pre_outcome": 0.9,
            "released_planning": False,
            "planning_beats_cost": True,
            "cost_regret": 1.0,
            "planning_regret": 0.0,
            "routed_regret": 1.0,
        }
    }
    assert _deployment_summary(rows, "planning")["planning_release_coverage"] == 1.0
    assert _deployment_summary(rows, "cost")["planning_release_coverage"] == 0.0


def test_post_execution_labels_are_identity_bound_and_include_only_realized_outcomes() -> None:
    labels = _post_execution_labels(
        [{"instance_id": "outcome-root", "candidate_id": 4, "goal_raw": [1.0, 0.0]}],
        np.asarray([[0.0, 0.0, 0.2, -0.3]]),
    )
    assert set(labels) == {"outcome-root/4"}
    assert labels["outcome-root/4"]["real_task_cost"] == 1.0
    assert labels["outcome-root/4"]["realized_terminal_state"] == [0.0, 0.0, 0.2, -0.3]


def test_execution_boundary_rejects_any_cross_role_root_reuse() -> None:
    train = [{**_row("candidate-0", 0.0), "instance_id": "shared-root"}]
    calibration = [{**_row("candidate-1", 0.0), "instance_id": "calibration-root"}]
    outcome = [{**_row("candidate-2", 0.0), "instance_id": "shared-root"}]
    with pytest.raises(ValueError, match="must be disjoint"):
        _validate_root_partition(train, calibration, outcome)


def test_execution_boundary_reports_unique_root_counts() -> None:
    partitions = _validate_root_partition(
        [{**_row("candidate-0", 0.0), "instance_id": "train-root"}],
        [{**_row("candidate-1", 0.0), "instance_id": "calibration-root"}],
        [{**_row("candidate-2", 0.0), "instance_id": "outcome-root"}],
    )
    assert partitions == {"train": 1, "calibration": 1, "outcome": 1}


def test_v7_execution_boundary_requires_proven_shared_candidate_support(tmp_path) -> None:
    ledger = tmp_path / "preflight.json"
    ledger.write_text(json.dumps({
        "schema": "pointmaze_planning_shaped_advantage_v7_static_preflight.v1",
        "status": "passed_static_preflight",
        "post_execution_aliasing_contract": {
            "representation_key": "source_history_model",
            "radius_quantile": 0.25,
            "contrast_gap_threshold": 0.05,
            "max_reported_pairs": 64,
        },
        "candidate_bank": {
            "shared_candidate_in_support": True,
            "held_action_hash_count": 1,
            "held_train_action_hash_overlap_count": 1,
            "scorer_candidate_bank_protocol": "held_exact_shared_bank",
            "scorer_training_action_hash_count": 1,
            "oracle_headroom_contract": "design_root_terminal_goal_cost_v1",
            "design_best_second_cost_gap_q10": 0.001,
            "design_best_second_cost_gap_ge_0p003_fraction": 0.50,
            "design_candidate_cost_spread_q10": 0.05,
            "router_feature_contract": "full_ordered_20x2_root_features_plus_frozen_effect_disagreement_v1",
            "router_feature_dimension": 95,
            "router_train_root_count": 111,
            "minimum_router_train_root_surplus": 16,
            "router_kind": "advantage_ridge",
            "router_action_representation": "full_ordered_20x2",
            "router_regularization_selection": "train_root_only_four_fold_blocked_advantage_mse",
        },
    }), encoding="utf-8")
    source = {"records": [
        {"candidate_action_hash": "a", "instance_id": f"train-{index}", "split": "train"}
        for index in range(111)
    ]}
    context = {"metadata": {
        "candidate_generalization_regime": "shared_candidate_in_support",
        "candidate_design": "goal_balanced_interaction_pairs",
        "scorer_candidate_bank_protocol": "held_exact_shared_bank",
        "frozen_preflight_ledger": {
            "schema": "pointmaze_planning_shaped_advantage_v7_static_preflight.v1",
            "status": "passed_static_preflight", "path": str(ledger),
        },
    }, "records": [{"candidate_action_hash": "a"}]}
    result = _validate_shared_candidate_support(source, context)
    assert result["verified"] is True
    with pytest.raises(ValueError, match="advantage_ridge"):
        _validate_shared_candidate_support(source, context, router_kind="logistic")
    with pytest.raises(ValueError, match="advantage_ridge"):
        _validate_shared_candidate_support(source, context, action_representation="summary_mean_std_first_last")
    source["records"].append({"candidate_action_hash": "extra"})
    with pytest.raises(ValueError, match="exactly identical"):
        _validate_shared_candidate_support(source, context)
    source["records"] = [{"candidate_action_hash": "a"}]
    context["records"] = [{"candidate_action_hash": "missing"}]
    with pytest.raises(ValueError, match="exactly identical"):
        _validate_shared_candidate_support(source, context)
