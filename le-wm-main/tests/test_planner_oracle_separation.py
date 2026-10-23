from __future__ import annotations

import json

import pytest

from experiments.audit_planner_oracle_separation import (
    _identity_hash,
    build_audit,
    load_spec,
    write_audit,
)


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _artifact(path, method, success_rate, *, cohort="fresh-roots-v1", bank="bank-v1", horizon=4, roots=None, actions=None):
    roots = ["root-1", "root-2"] if roots is None else roots
    actions = ["action-hash-1", "action-hash-2"] if actions is None else actions
    return _write(path, {
        "schema": "test.result.v1",
        "protocol": {
            "methods": [method],
            "cohort_id": cohort,
            "candidate_bank_id": bank,
            "horizon": horizon,
            "task_objective_id": "distance-to-goal-v1",
            "outcome_root_ids": roots,
            "outcome_root_ids_sha256": _identity_hash(roots),
            "candidate_action_hashes": actions,
            "candidate_action_hashes_sha256": _identity_hash(actions),
            "candidate_action_count": len(actions),
            "candidate_bank_mode": "shared_fixed",
            "execution_semantics": "receding_horizon_h4_fixed_bank",
        },
        "aggregates": {method: {"success_rate": success_rate}},
    })


def _spec(tmp_path, arms):
    return _write(tmp_path / "spec.json", {
        "schema": "planner_oracle_separation_spec.v2",
        "contract": {
            "cohort_id": "fresh-roots-v1",
            "candidate_bank_id": "bank-v1",
            "horizon": 4,
            "task_objective_id": "distance-to-goal-v1",
            "metric_direction": "higher_is_better",
            "minimum_success_rate": 0.5,
            "outcome_root_ids_sha256": _identity_hash(["root-1", "root-2"]),
            "candidate_action_hashes_sha256": _identity_hash(["action-hash-1", "action-hash-2"]),
            "candidate_action_count": 2,
            "execution_semantics": "receding_horizon_h4_fixed_bank",
        },
        "arms": arms,
    })


def _arm(path, method):
    return {
        "artifact": path.name,
        "declared_method": method,
        "metric_path": f"aggregates.{method}.success_rate",
    }


def test_candidate_ceiling_failure_routes_without_reading_other_outcomes(tmp_path):
    candidate = _artifact(tmp_path / "candidate.json", "ground_truth_candidate_oracle", 0.25)
    spec = _spec(tmp_path, {"candidate_oracle": _arm(candidate, "ground_truth_candidate_oracle")})

    report = build_audit(load_spec(spec))

    assert report["route"]["label"] == "candidate_or_horizon_limited"
    assert report["route"]["status"] == "identified"
    assert report["selection_policy"]["test_outcome_model_selection"] is False
    assert report["claim_boundary"]["does_not_establish"]


def test_real_successor_learned_geometry_failure_routes_to_ranking(tmp_path):
    candidate = _artifact(tmp_path / "candidate.json", "ground_truth_candidate_oracle", 1.0)
    perfect = _artifact(tmp_path / "perfect.json", "perfect_dynamics_mpc", 1.0)
    geometry = _artifact(tmp_path / "geometry.json", "real_successor_learned_geometry", 0.25)
    spec = _spec(tmp_path, {
        "candidate_oracle": _arm(candidate, "ground_truth_candidate_oracle"),
        "perfect_dynamics_mpc": _arm(perfect, "perfect_dynamics_mpc"),
        "real_successor_learned_geometry": _arm(geometry, "real_successor_learned_geometry"),
    })

    report = build_audit(load_spec(spec))

    assert report["route"]["label"] == "ranking_limited"
    assert report["route"]["required_controls_missing"] == []


def test_oracle_readout_failure_routes_to_information_support(tmp_path):
    candidate = _artifact(tmp_path / "candidate.json", "ground_truth_candidate_oracle", 1.0)
    perfect = _artifact(tmp_path / "perfect.json", "perfect_dynamics_mpc", 1.0)
    geometry = _artifact(tmp_path / "geometry.json", "real_successor_learned_geometry", 1.0)
    dynamics = _artifact(tmp_path / "dynamics.json", "learned_dynamics_oracle_readout", 0.25)
    spec = _spec(tmp_path, {
        "candidate_oracle": _arm(candidate, "ground_truth_candidate_oracle"),
        "perfect_dynamics_mpc": _arm(perfect, "perfect_dynamics_mpc"),
        "real_successor_learned_geometry": _arm(geometry, "real_successor_learned_geometry"),
        "learned_dynamics_oracle_readout": _arm(dynamics, "learned_dynamics_oracle_readout"),
    })

    report = build_audit(load_spec(spec))

    assert report["route"]["label"] == "information_or_support_limited"


def test_missing_mixed_controls_fails_closed_instead_of_overcalling_ranking(tmp_path):
    candidate = _artifact(tmp_path / "candidate.json", "ground_truth_candidate_oracle", 1.0)
    perfect = _artifact(tmp_path / "perfect.json", "perfect_dynamics_mpc", 1.0)
    spec = _spec(tmp_path, {
        "candidate_oracle": _arm(candidate, "ground_truth_candidate_oracle"),
        "perfect_dynamics_mpc": _arm(perfect, "perfect_dynamics_mpc"),
    })

    report = build_audit(load_spec(spec))

    assert report["route"]["label"] == "unidentified_missing_oracle_controls"
    assert report["route"]["required_controls_missing"] == [
        "real_successor_learned_geometry",
        "learned_dynamics_oracle_readout",
    ]


def test_contract_mismatch_fails_closed_and_output_is_machine_readable(tmp_path):
    candidate = _artifact(tmp_path / "candidate.json", "ground_truth_candidate_oracle", 1.0)
    geometry = _artifact(
        tmp_path / "geometry.json", "real_successor_learned_geometry", 0.25,
        bank="different-bank",
    )
    spec = _spec(tmp_path, {
        "candidate_oracle": _arm(candidate, "ground_truth_candidate_oracle"),
        "real_successor_learned_geometry": _arm(geometry, "real_successor_learned_geometry"),
    })

    report = build_audit(load_spec(spec))
    output = tmp_path / "audit.json"
    write_audit(report, output)

    assert report["route"]["label"] == "protocol_mismatch"
    assert report["protocol_validation"]["valid"] is False
    assert json.loads(output.read_text(encoding="utf-8"))["schema"] == "planner_oracle_separation.v2"


def test_spec_rejects_ambiguous_metric_pointer_and_unlisted_method(tmp_path):
    candidate = _artifact(tmp_path / "candidate.json", "ground_truth_candidate_oracle", 1.0)
    spec = _spec(tmp_path, {
        "candidate_oracle": {
            "artifact": candidate.name,
            "declared_method": "unlisted_method",
            "metric_path": "aggregates.*.success_rate",
        },
    })

    with pytest.raises(ValueError, match="wildcards|declared_method"):
        build_audit(load_spec(spec))


def test_same_named_bank_with_different_action_identity_fails_closed(tmp_path):
    candidate = _artifact(tmp_path / "candidate.json", "ground_truth_candidate_oracle", 1.0)
    perfect = _artifact(
        tmp_path / "perfect.json", "perfect_dynamics_mpc", 1.0,
        actions=["action-hash-1", "different-action-hash"],
    )
    spec = _spec(tmp_path, {
        "candidate_oracle": _arm(candidate, "ground_truth_candidate_oracle"),
        "perfect_dynamics_mpc": _arm(perfect, "perfect_dynamics_mpc"),
    })

    report = build_audit(load_spec(spec))

    assert report["route"]["label"] == "protocol_mismatch"
    assert "perfect_dynamics_mpc" in report["protocol_validation"]["mismatches"]
