from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.gdd_wm import run_pointmaze_counterfactual_endpoint_control as endpoint_control
from experiments.gdd_wm.run_pointmaze_planning_shaped_control import action_representation_input_fields
from experiments.gdd_wm.run_pointmaze_counterfactual_endpoint_control import _assert_action_summary_collision_contract, _assert_split_contract, _assert_summary_prediction_alias_contract, _calibration_role, _enforce_arm_contract, _endpoint_ensemble_diagnostics, _factorial_summary, _policy, _root_fold, _summary


def _row(*, cost: float, endpoint: float, planning: float, disagree: bool) -> dict:
    return {
        "cost_regret": cost,
        "endpoint_regret": endpoint,
        "planning_regret": planning,
        "summary_cost_regret": cost,
        "summary_endpoint_regret": endpoint,
        "endpoint_disagrees_with_cost_pre_outcome": disagree,
        "endpoint_beats_cost": endpoint < cost,
    }


def _metadata(*, candidate_design: str = "goal_balanced_interaction_pairs", entropy: float = 0.75) -> dict:
    return {
        "candidate_generalization_regime": "shared_candidate_in_support",
        "candidate_design": candidate_design,
        "interaction_pair_design": {"design_winner_entropy_normalized": entropy},
        "frozen_preflight_ledger": {
            "schema": "pointmaze_counterfactual_endpoint_v6_static_preflight.v1",
            "status": "passed_static_preflight", "sha256": "a" * 64,
            "context_count": 3, "root_counts": {"train": 1, "calibration": 1, "outcome": 1},
        },
    }


def test_action_representation_schema_is_explicit_and_closed_under_v6_bases() -> None:
    assert action_representation_input_fields("full_ordered_20x2") == [
        "source_history_model", "goal_raw", "action_sequence",
    ]
    assert action_representation_input_fields("canonical_summary_mean_std_first_last") == [
        "source_history_model", "goal_raw", "action_sequence_summary",
    ]
    with pytest.raises(ValueError, match="unknown action representation"):
        action_representation_input_fields("unrecognized_basis")


def test_v6_arm_contract_rejects_missing_or_crossed_action_schema() -> None:
    ordered = {
        "action_representation": "full_ordered_20x2",
        "input_fields": ["source_history_model", "goal_raw", "action_sequence"],
    }
    _enforce_arm_contract(ordered, expected_representation="full_ordered_20x2")
    with pytest.raises(ValueError, match="actual=None"):
        _enforce_arm_contract({}, expected_representation="full_ordered_20x2")
    crossed = {**ordered, "input_fields": ["source_history_model", "goal_raw", "action_sequence_summary"]}
    with pytest.raises(ValueError, match="incompatible input fields"):
        _enforce_arm_contract(crossed, expected_representation="full_ordered_20x2")


def test_endpoint_summary_separates_all_roots_from_action_disagreement_stratum() -> None:
    value = _summary({
        "same": _row(cost=0.0, endpoint=0.0, planning=0.0, disagree=False),
        "different": _row(cost=1.0, endpoint=0.0, planning=0.5, disagree=True),
    })
    assert value["endpoint_disagreement_count"] == 1
    assert value["disagreement_stratum"]["mean_endpoint_minus_cost_regret"] == -1.0
    assert value["mean_regret"]["endpoint"] == 0.0


def test_factorial_summary_reports_representation_readout_interaction_without_selecting_an_arm() -> None:
    value = _factorial_summary({
        "root": {
            **_row(cost=0.3, endpoint=0.1, planning=0.2, disagree=True),
            "summary_cost_regret": 0.5, "summary_endpoint_regret": 0.4,
        },
    })
    assert value["ordered_minus_summary_cost_regret"] == pytest.approx(-0.2)
    assert value["endpoint_minus_cost_ordered_regret"] == pytest.approx(-0.2)
    assert value["readout_by_representation_interaction"] == pytest.approx(-0.1)


def test_endpoint_policy_requires_selector_and_certification_disagreement_support() -> None:
    rows = {}
    role_counts = {0: 0, 1: 0, 2: 0}
    # Populate each hash-partitioned calibration role with 12 roots.
    index = 0
    while min(role_counts.values()) < 12:
        root = f"root-{index}"
        role = _calibration_role(root)
        if role_counts[role] < 12:
            rows[root] = _row(cost=1.0, endpoint=0.0, planning=0.5, disagree=True)
            role_counts[role] += 1
        index += 1
    policy = _policy(rows, seed=17, minimum_disagreement_roots=12)
    assert policy["endpoint_selected"] is True
    assert policy["endpoint_certified"] is True
    assert policy["deployment_policy"] == "endpoint"


def test_endpoint_policy_fails_closed_when_disagreement_stratum_has_insufficient_support() -> None:
    rows, counts, index = {}, {0: 0, 1: 0, 2: 0}, 0
    while min(counts.values()) < 8:
        root = f"sparse-{index}"
        role = _calibration_role(root)
        if counts[role] < 8:
            rows[root] = _row(cost=1.0, endpoint=0.0, planning=0.5, disagree=index < 2)
            counts[role] += 1
        index += 1
    policy = _policy(rows, seed=17, minimum_disagreement_roots=12)
    assert policy["endpoint_selected"] is False
    assert policy["endpoint_certified"] is False
    assert policy["deployment_policy"] == "cost"


def test_split_contract_rejects_root_reuse_and_outcome_label_leakage() -> None:
    train = [{"instance_id": "train"}]
    calibration = [{"instance_id": "calibration"}]
    outcome = [{"instance_id": "outcome", "realized_terminal_state": [0.0, 0.0, 0.0, 0.0]}]
    with np.testing.assert_raises_regex(ValueError, "terminal labels"):
        _assert_split_contract(train, calibration, outcome, _metadata())
    with np.testing.assert_raises_regex(ValueError, "root split overlap"):
        _assert_split_contract(train, [{"instance_id": "train"}], [{"instance_id": "outcome"}], _metadata())
    with np.testing.assert_raises_regex(ValueError, "goal_balanced"):
        _assert_split_contract(train, calibration, [{"instance_id": "outcome"}], _metadata(candidate_design="state_interaction_pairs"))
    with np.testing.assert_raises_regex(ValueError, "winner-diversity"):
        _assert_split_contract(train, calibration, [{"instance_id": "outcome"}], _metadata(entropy=0.49))
    stale = _metadata(); stale["frozen_preflight_ledger"]["root_counts"]["outcome"] = 99
    with np.testing.assert_raises_regex(ValueError, "root counts"):
        _assert_split_contract(train, calibration, [{"instance_id": "outcome"}], stale)


def test_action_summary_collision_contract_requires_distinct_sequences_with_identical_legacy_features() -> None:
    first = np.repeat(np.asarray([[0.5, 0.0], [0.1, 0.4], [-0.3, 0.2], [0.2, -0.4], [0.4, 0.1]]), 4, axis=0)
    second = np.repeat(np.asarray([[0.5, 0.0], [-0.3, 0.2], [0.1, 0.4], [0.2, -0.4], [0.4, 0.1]]), 4, axis=0)
    rows = [
        {"instance_id": "root", "candidate_id": "left", "action_sequence": first.tolist()},
        {"instance_id": "root", "candidate_id": "right", "action_sequence": second.tolist()},
    ]
    result = _assert_action_summary_collision_contract(rows, rows, rows, minimum_pairs_per_root=1)
    assert result["checked_root_count"] == 3
    assert all(value["summary_collision_pair_count"] == 1 for value in result["per_root"].values())
    with np.testing.assert_raises_regex(ValueError, "summary-colliding"):
        _assert_action_summary_collision_contract(
            rows,
            [{**rows[0], "action_sequence": first.tolist()}, {**rows[1], "action_sequence": first.tolist()}],
            rows,
            minimum_pairs_per_root=1,
        )


def test_summary_prediction_alias_contract_rejects_a_downstream_side_channel() -> None:
    first = np.repeat(np.asarray([[0.5, 0.0], [0.1, 0.4], [-0.3, 0.2], [0.2, -0.4], [0.4, 0.1]]), 4, axis=0)
    second = np.repeat(np.asarray([[0.5, 0.0], [-0.3, 0.2], [0.1, 0.4], [0.2, -0.4], [0.4, 0.1]]), 4, axis=0)
    rows = [{
        "instance_id": "root", "candidate_id": candidate, "source_history_model": [0.1, 0.2],
        "goal_raw": [1.0, 0.0], "action_sequence": action.tolist(),
    } for candidate, action in (("left", first), ("right", second))]
    constant = lambda feature: (np.zeros(len(feature)), np.zeros((len(feature), 4)))
    endpoint_constant = lambda feature, goal: (np.zeros(len(feature)), np.zeros((len(feature), 4)))
    result = _assert_summary_prediction_alias_contract({"root": rows}, constant, endpoint_constant)
    assert result["per_root"]["root"]["summary_collision_pair_count"] == 1
    leaking = lambda feature: (np.arange(len(feature), dtype=np.float64), np.zeros((len(feature), 4)))
    with np.testing.assert_raises_regex(ValueError, "violate canonical"):
        _assert_summary_prediction_alias_contract({"root": rows}, leaking, endpoint_constant)
    malformed = lambda feature: (np.zeros(len(feature) - 1), np.zeros((len(feature), 4)))
    with np.testing.assert_raises_regex(ValueError, "one score per candidate"):
        _assert_summary_prediction_alias_contract({"root": rows}, malformed, endpoint_constant)


def test_endpoint_ensemble_diagnostics_is_pre_outcome_and_does_not_change_deployed_score() -> None:
    rows = [{
        "instance_id": "root", "candidate_id": str(index), "source_history_model": [0.1, 0.2],
        "goal_raw": [1.0, 0.0], "action_sequence": np.zeros((20, 2)).tolist(),
    } for index in range(2)]

    def predictor(features, goals):
        endpoint = np.asarray([[0.2, 0.0, 0.0, 0.0], [0.8, 0.0, 0.0, 0.0]])
        return np.linalg.norm(endpoint[:, :2] - goals, axis=1), endpoint

    def members(_features, _goals):
        return np.asarray([
            [[0.2, 0.0, 0.0, 0.0], [0.8, 0.0, 0.0, 0.0]],
            [[0.4, 0.0, 0.0, 0.0], [0.7, 0.0, 0.0, 0.0]],
            [[0.1, 0.0, 0.0, 0.0], [0.9, 0.0, 0.0, 0.0]],
        ])

    setattr(predictor, "member_endpoint_predictions", members)
    result = _endpoint_ensemble_diagnostics({"root": rows}, predictor)
    assert result["available"] is True
    assert result["per_root"]["root"]["deployed_candidate_id"] == "1"
    assert result["per_root"]["root"]["member_agreement_with_deployed_choice"] == pytest.approx(1.0)
    assert "does not calibrate" in result["claim_boundary"]


def test_runner_serializes_outcomes_only_after_frozen_endpoint_policy(tmp_path, monkeypatch) -> None:
    """Exercise the output/split contract without treating a CPU fixture as evidence."""

    def row(root: str, candidate: int, split: str, terminal: list[float]) -> dict:
        pulses = np.asarray([
            [0.5, 0.0], [0.1, 0.4], [-0.3, 0.2], [0.2, -0.4], [0.4, 0.1],
        ], dtype=np.float64)
        if candidate == 1:
            pulses[[1, 2]] = pulses[[2, 1]]
        return {
            "instance_id": root, "candidate_id": candidate, "split": split,
            "source_history_model": [0.1, 0.2, 0.3, 0.4], "source_latent_model": [0.3, 0.4],
            "source_state_raw": [0.0, 0.0, 0.0, 0.0], "goal_raw": [1.0, 0.0],
            "action_sequence": np.repeat(pulses, 4, axis=0).tolist(),
            "synthetic_terminal": terminal,
        }

    roots_by_role = {0: [], 1: [], 2: []}
    index = 0
    while min(len(value) for value in roots_by_role.values()) < 12:
        root = f"cal-{index}"
        role = _calibration_role(root)
        if len(roots_by_role[role]) < 12:
            roots_by_role[role].append(root)
        index += 1
    calibration_rows = [row(root, candidate, "calibration", [float(candidate), 0.0, 0.0, 0.0]) for roots in roots_by_role.values() for root in roots for candidate in (0, 1)]
    outcome_rows = [row(f"out-{index}", candidate, "outcome", [float(candidate), 0.0, 0.0, 0.0]) for index in range(2) for candidate in (0, 1)]
    train_rows = [
        {**row("train", candidate, "train", [float(candidate), 0.0, 0.0, 0.0]), "realized_terminal_state_ref": f"arrays.npz::terminal_{candidate}"}
        for candidate in (0, 1)
    ]
    source = tmp_path / "source.json"
    arrays = tmp_path / "arrays.npz"
    context = tmp_path / "context.json"
    calibration = tmp_path / "calibration.json"
    output = tmp_path / "output"
    source.write_text(json.dumps({"records": train_rows}), encoding="utf-8")
    np.savez(arrays, terminal_0=np.asarray([0.0, 0.0, 0.0, 0.0]), terminal_1=np.asarray([1.0, 0.0, 0.0, 0.0]))
    metadata = _metadata()
    metadata["frozen_preflight_ledger"]["root_counts"] = {
        "train": 1, "calibration": len(calibration_rows) // 2, "outcome": len(outcome_rows) // 2,
    }
    context.write_text(json.dumps({
        "metadata": metadata,
        "records": calibration_rows + outcome_rows,
    }), encoding="utf-8")
    calibration.write_text(json.dumps({
        f"{root}/{candidate}": {"realized_terminal_state": [float(candidate), 0.0, 0.0, 0.0]}
        for roots in roots_by_role.values() for root in roots for candidate in (0, 1)
    }), encoding="utf-8")

    def direct_stub(*_args, **kwargs):
        representation = kwargs["action_representation"]
        summary = representation == "canonical_summary_mean_std_first_last"
        return {
            "objective": "stub", "action_representation": representation,
            "input_fields": action_representation_input_fields(representation),
        }, lambda feature: (
            np.zeros(len(feature)) if summary else np.tile([0.0, 1.0], len(feature) // 2),
            np.zeros((len(feature), 4)),
        )

    def endpoint_stub(*_args, **kwargs):
        representation = kwargs["action_representation"]
        summary = representation == "canonical_summary_mean_std_first_last"
        def predict(feature, goal):
            endpoint = np.zeros((len(feature), 4))
            if not summary:
                endpoint[1::2, 0] = 1.0
            return np.zeros(len(feature)) if summary else np.tile([1.0, 0.0], len(feature) // 2), endpoint
        return {
            "objective": "endpoint_stub", "action_representation": representation,
            "input_fields": action_representation_input_fields(representation),
        }, predict

    calls: list[str] = []
    monkeypatch.setattr(endpoint_control, "require_cuda_allocation", lambda _device: {"device": "stub-cuda"})
    monkeypatch.setattr(endpoint_control, "_fit_ensemble", direct_stub)
    monkeypatch.setattr(endpoint_control, "_fit_endpoint_ensemble", endpoint_stub)
    monkeypatch.setattr(endpoint_control, "_fresh_terminal", lambda rows: (calls.append("outcome") or np.asarray([value["synthetic_terminal"] for value in rows], dtype=np.float64)))
    result = endpoint_control.run(
        source_dataset=source, source_arrays=arrays, context=context, calibration_labels=calibration,
        output=output, device="cuda", ensemble_size=1, steps=1,
        minimum_summary_collision_pairs_per_root=1,
    )
    assert calls == ["outcome"]
    assert result["calibration"]["policy"]["endpoint_certified"] is True
    assert result["outcome"]["authorized_deployment_policy"] == "endpoint"
    assert result["control_contract"]["action_representation"] == "full_ordered_20x2"
    assert result["arms"]["cost_only"]["action_representation"] == "full_ordered_20x2"
    assert result["arms"]["counterfactual_endpoint"]["action_representation"] == "full_ordered_20x2"
    assert result["arms"]["planning_shaped_auxiliary"]["input_fields"] == [
        "source_history_model", "goal_raw", "action_sequence",
    ]
    assert result["arms"]["summary_cost_only"]["action_representation"] == "canonical_summary_mean_std_first_last"
    assert result["arms"]["summary_counterfactual_endpoint"]["input_fields"] == [
        "source_history_model", "goal_raw", "action_sequence_summary",
    ]
    assert result["control_contract"]["factorial_diagnostic"]["selection_uses_ordered_endpoint_vs_ordered_cost_only"] is True
    assert result["summary_prediction_aliasing_pre_outcome"]["outcome"]["checked_before_outcome_execution"] is True
    assert result["post_repair_contrast_certificate"]["released_root_count"] == 2
    assert len(result["post_execution_outcome_labels"]) == 4
    serialized = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert serialized["post_execution_outcome_labels"] == result["post_execution_outcome_labels"]
