from copy import deepcopy

import pytest

from experiments.spectral_reachability.d2p_p1_p2_p3_interface_contract_20261102_codexexclusive import (
    validate,
)


def identity():
    return {
        "root_id": "root-1900",
        "task_id": "task-family-a",
        "protocol_id": "protocol-v1",
        "frozen_model_id": "checkpoint-v1",
        "checkpoint_sha256": "1" * 64,
        "task_manifest_sha256": "2" * 64,
        "protocol_sha256": "3" * 64,
        "source_sha256": "4" * 64,
    }


def p1():
    return {
        "schema": "d2p_p1_admission_contract.v1",
        "status": "contract_validated",
        "identity": identity(),
        "P1_admission": False,
        "P2_admission": False,
        "P3_admission": False,
        "outcome_sealed_before_selection": True,
        "source_disjoint_outcome": True,
        "evidence": {
            "non_greedy_prefix": True,
            "own_state_capability": True,
            "closed_loop_outcome": True,
        },
        "outcome_rows": [{"root_id": "root-1900", "status": "completed"}],
        "strong_baseline_controls": ["greedy", "zero_prefix", "cem"],
    }


def p2():
    return {
        "schema": "d2p_p2_composition_search_contract.v1",
        "status": "contract_validated",
        "identity": identity(),
        "P1_admission": False,
        "P2_admission": False,
        "P3_admission": False,
        "candidate_library_sealed_before_outcome": True,
        "search_executed": True,
        "same_budget_and_feedback": True,
        "failed_or_partial_retained": True,
        "edges": [
            {
                "edge_id": "e0",
                "input_context": {"rgb": "before"},
                "predicted_postcondition": {"rgb": "predicted"},
                "observed_context": {"rgb": "observed"},
                "action_witness": [[0.1, 0.0]],
                "replay_receipt": {"status": "exact"},
                "observation_required": True,
                "recompiled_after_observation": True,
                "status": "recompiled",
            }
        ],
        "comparator": {"no_composition": True, "flat_search": True},
    }


def p3():
    return {
        "schema": "d2p_p3_dynamic_schedule_contract.v1",
        "status": "contract_validated",
        "identity": identity(),
        "P1_admission": False,
        "P2_admission": False,
        "P3_admission": False,
        "schedule_sealed_before_outcome": True,
        "same_feedback_and_budget_cap": True,
        "allocation_before_outcome": True,
        "signal": {
            "source": "model_dynamics",
            "read_after_outcome": False,
            "horizon": 3.0,
            "depth": 2.0,
            "predict_budget": 16.0,
        },
        "allocation": {"planned": 16, "actual": 12, "unused_budget": 4},
        "failed_work_refunded": False,
        "unknown_work_is_zero": False,
        "controls": {
            "fixed_horizon": True,
            "fixed_budget": True,
            "randomized_schedule": True,
        },
    }


@pytest.mark.parametrize(
    ("kind", "factory", "field"),
    [
        ("p1", p1, "evidence"),
        ("p2", p2, "edges"),
        ("p3", p3, "allocation"),
    ],
)
def test_valid_contracts_are_admission_neutral(kind, factory, field):
    payload = factory()
    result = validate(kind, payload)
    assert result == payload
    assert all(result[key] is False for key in ("P1_admission", "P2_admission", "P3_admission"))
    assert field in result


def test_p1_requires_same_root_three_way_evidence_and_source_disjoint_outcome():
    payload = p1()
    payload["evidence"]["own_state_capability"] = False
    with pytest.raises(ValueError, match="three same-root"):
        validate("p1", payload)

    payload = p1()
    payload["source_disjoint_outcome"] = False
    with pytest.raises(ValueError, match="source-disjoint"):
        validate("p1", payload)


def test_p2_rejects_replay_only_edges_and_missing_recompile():
    payload = p2()
    payload["search_executed"] = False
    with pytest.raises(ValueError, match="search"):
        validate("p2", payload)

    payload = p2()
    payload["edges"][0]["recompiled_after_observation"] = False
    with pytest.raises(ValueError, match="recompile"):
        validate("p2", payload)


def test_p2_retains_failures_and_requires_both_controls():
    payload = p2()
    payload["failed_or_partial_retained"] = False
    with pytest.raises(ValueError, match="censor"):
        validate("p2", payload)

    payload = p2()
    payload["comparator"]["flat_search"] = False
    with pytest.raises(ValueError, match="controls"):
        validate("p2", payload)


def test_p3_rejects_outcome_dependent_schedule_and_hidden_budget():
    payload = p3()
    payload["signal"]["read_after_outcome"] = True
    with pytest.raises(ValueError, match="must not read outcome"):
        validate("p3", payload)

    payload = p3()
    payload["unknown_work_is_zero"] = True
    with pytest.raises(ValueError, match="unknown work"):
        validate("p3", payload)

    payload = p3()
    payload["allocation"]["unused_budget"] = 0
    with pytest.raises(ValueError, match="arithmetic remainder"):
        validate("p3", payload)


def test_admission_positive_or_historical_gate_is_rejected():
    for kind, factory, field in (("p1", p1, "P1_admission"),
                                 ("p2", p2, "P2_admission"),
                                 ("p3", p3, "P3_admission")):
        payload = deepcopy(factory())
        payload[field] = True
        with pytest.raises(ValueError, match="must remain false"):
            validate(kind, payload)


def test_identity_and_missing_action_witness_fail_closed():
    payload = p1()
    payload["identity"]["checkpoint_sha256"] = "not-a-sha"
    with pytest.raises(ValueError, match="SHA256"):
        validate("p1", payload)

    payload = p2()
    del payload["edges"][0]["action_witness"]
    with pytest.raises(ValueError, match="action_witness"):
        validate("p2", payload)
