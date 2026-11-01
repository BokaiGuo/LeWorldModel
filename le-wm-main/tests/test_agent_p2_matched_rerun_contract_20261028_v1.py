"""RAM budgets/receipt mutations and saved-only task closure. No neural imports."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.spectral_reachability import agent_p2_matched_rerun_contract_20261028_v1 as m


def protocols():
    return {"kinodynamic": {"contract": {"schedule": dict(first20=20, beam_width=6, second120=120,
        planning_predict=140, fresh_whole_predict=2, fresh_segmented_predict=2, total_predict=144, high_level_dispatches=143)}},
        "no_composition": {"contract": {"per_root_budget": dict(predict=88, action_encoder=88,
            jacobian=1, vjp_directions=192, svd=2, bounded_solve=1),
            "calls": dict(nominal=2, AD_forward=2, candidates=80, winner_whole=2, winner_segmented=2)}}}


def test_preserve_native_not_equalized():
    vectors = m.native_vectors(protocols())
    assert vectors["kinodynamic"]["predict_API"]["limit"] == 144
    assert vectors["no_composition"]["predict_API"]["limit"] == 88
    assert vectors["no_composition"]["AD_forward"]["limit"] == 2
    assert vectors["no_composition"]["VJP"]["limit"] == 192
    assert vectors["no_composition"]["J"]["limit"] == 1
    assert vectors["no_composition"]["SVD"]["limit"] == 2
    assert vectors["no_composition"]["solve"]["limit"] == 1
    assert vectors["kinodynamic"]["wall_seconds"]["limit"] is None
    assert vectors["no_composition"]["predict_API"]["inherited"] is None
    assert vectors["no_composition"]["projector"]["inherited"] == 4


@pytest.mark.parametrize("mutation", ["144to88", "spectral144", "extra_replay", "omitAD"])
def test_changed_budget_refused(mutation):
    p = protocols()
    if mutation == "144to88":
        p["kinodynamic"]["contract"]["schedule"]["total_predict"] = 88
    elif mutation == "spectral144":
        p["no_composition"]["contract"]["per_root_budget"]["predict"] = 144
    else:
        p["no_composition"]["contract"]["calls"]["AD_forward" if mutation == "omitAD" else "winner_whole"] = 0 if mutation == "omitAD" else 4
    with pytest.raises(ValueError):
        m.native_vectors(p)


def decision():
    return dict(schema=m.NAME + ".decision", budget_policy="retain_native_144_vs_88",
        acknowledge_unequal_API_AD=True, wall_policy="retain_original_scopes_no_new_wall_claim")


def test_no_choice_not_silently_selected():
    assert m.decision_status(None)["blocked"]
    assert not m.decision_status(decision())["blocked"]


@pytest.mark.parametrize("policy", ["equal144", "equal88", "common144", "effect_selected_budget"])
def test_nonexisting_equal_protocol_refused(policy):
    d = decision()
    d["budget_policy"] = policy
    with pytest.raises(ValueError, match="unsupported"):
        m.decision_status(d)


def test_no_implicit_acknowledgement():
    d = decision()
    d["acknowledge_unequal_API_AD"] = False
    with pytest.raises(ValueError, match="acknowledgement"):
        m.decision_status(d)


def requirement_and_receipt():
    requirement = m.loader_requirements({"/weights": "w", "/config": "c"}, "t")
    receipt = {"schema": requirement["schema"], "run_id": "MOCK_NEW_RUN", "contract_sha256": "p",
        "loader_source_sha256": requirement["loader_source"]["sha256"], "device": "cuda:0",
        "files": [{"path": p, "sha256": h, "bytes": 123, "hash_buffer_sha256": h,
                   "deserialize_buffer_sha256": h, "same_buffer": True} for p, h in requirement["expected_files"].items()],
        "tensor_hash_before": "t", "tensor_hash_after": "t", "frozen_eval": True, "errors": []}
    return requirement, receipt


def test_mock_schema_not_old_run_proof():
    requirement, receipt = requirement_and_receipt()
    out = m.validate_loader_receipt(receipt, requirement, "p")
    assert not out["old_runs_recertified"]
    assert out["status"] == "receipt_schema_bound_not_independent_runtime_proof"


@pytest.mark.parametrize("mutation", ["reopen", "differentbytes", "cpu", "tensordrift", "missingfile", "duplicates", "contract"])
def test_loader_mutations(mutation):
    requirement, receipt = requirement_and_receipt()
    if mutation == "reopen":
        receipt["files"][0]["same_buffer"] = False
    elif mutation == "differentbytes":
        receipt["files"][0]["deserialize_buffer_sha256"] = "other"
    elif mutation == "cpu":
        receipt["device"] = "cpu"
    elif mutation == "tensordrift":
        receipt["tensor_hash_after"] = "other"
    elif mutation == "missingfile":
        receipt["files"].pop()
    elif mutation == "duplicates":
        receipt["files"][1] = deepcopy(receipt["files"][0])
    else:
        receipt["contract_sha256"] = "old"
    with pytest.raises(ValueError):
        m.validate_loader_receipt(receipt, requirement, "p")


def test_existing_cli_functions_ast_only():
    plan = m.command_plan("kinodynamic")
    assert plan["callables"]["run"]["positional_parameters"] == ["prepared", "expected", "output"]
    assert plan["argv_templates"][2][-2:] == ["--output", "NEW_kinodynamic_CUDA_OUTPUT"]
    assert not plan["launchable_under_this_contract"]


def test_actual_saved_contract_no_future_execution_claim():
    p = m.build()
    assert len(p["tasks"]) == 2 and len(p["fixed_rows"]) == 4
    assert {r["arm"] for r in p["fixed_rows"]} == set(m.ARMS.values())
    assert p["decision"]["blocked"] and not p["run_authorized"]
    assert not p["old_runs_recertified"] and not p["equal_predict_calls"]
    assert p["inherited_shared_once"]["resources"]["encoder"]["completed"] == 8
    assert p["per_method_job_limits"]["no_composition"]["wall_seconds"] is None
    assert len(m.saved.canonical(p)) < m.PREP_CAP


def test_unbound_decision_refused():
    with pytest.raises(ValueError, match="external binding"):
        m.build(decision(), None)


def test_prepare_refuses_existing_output_before_overwrite(monkeypatch):
    monkeypatch.setattr(m, "build", lambda *a: {"schema": "MOCK"})
    with pytest.raises(ValueError, match="new bounded"):
        m.prepare(Path(__file__).absolute())
