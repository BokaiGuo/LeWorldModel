"""Pure logic/synthetic receipt tests, NOT CUDA or physical efficacy evidence."""
from copy import deepcopy
from dataclasses import asdict, fields, replace
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability.d2p_adaptive_budget_v2 import PredictionBudget
from experiments.spectral_reachability.d2p_p3_paired_accounting import (
    Binding, BranchReceipt, BudgetContract, PairedPlan, Resources, RootPlan,
    RunReceipt, compare_paired,
)


def resources(**changes):
    values = {f.name: 0 for f in fields(Resources)}
    values.update(changes)
    return Resources(**values)


def plan():
    caps = Resources(**{f.name: 100 for f in fields(Resources)})
    return PairedPlan("externally-presealed-test-only", tuple(
        RootPlan(BudgetContract(Binding("protocol", "task", root, "model"), caps),
                 ("branch-a", "branch-b")) for root in ("root-a", "root-b")
    ))


def run(root, method):
    binding = root.contract.binding
    # Failure includes attempted prediction, action/input backward and execution.
    branches = (
        BranchReceipt("branch-a", "completed", True, resources(
            predict_calls=1, predict_samples=3, rollout_samples=1,
            imagined_action_tokens=3, max_planning_horizon_tokens=3,
            physical_action_tokens=2, physical_env_steps=2, physical_branches=1)),
        BranchReceipt("branch-b", "failed", False, resources(
            predict_calls=1, predict_samples=2, action_backward_calls=2,
            action_vjp_rows=4, input_backward_calls=1, input_vjp_rows=3,
            physical_action_tokens=1, physical_env_steps=1, physical_branches=1)),
    )
    receipt = {"schema": "d2p_adaptive_budget.v2", "binding": asdict(binding),
               "method": method, "predict_sample_limit": 100,
               "attempted_predict_samples": 7, "blocked_calls": 1,
               "phases": [
                   {"phase": "planning", "method": method, "predict_calls": 2,
                    "predict_samples": 5, "failed_predict_calls": 1, "completed": False},
                   {"phase": "evaluation", "method": method, "predict_calls": 1,
                    "predict_samples": 2, "failed_predict_calls": 0, "completed": True},
               ]}
    return RunReceipt(binding, method, branches, resources(
        predict_calls=1, predict_samples=2, encoder_forwards=1, encoder_samples=2,
        max_planning_horizon_tokens=2), receipt)


def cohort():
    protocol = plan()
    return protocol, [run(r, "fixed") for r in protocol.roots], [
        run(r, "adaptive") for r in protocol.roots]


def test_identity_join_reversed_order_not_positional_zip_and_no_stage_claims():
    protocol, fixed, adaptive = cohort()
    adaptive[0] = replace(adaptive[0], overhead=replace(
        adaptive[0].overhead, action_backward_calls=8))
    result = compare_paired(protocol, fixed, adaptive[::-1])
    assert [p["binding"]["root_id"] for p in result["pairs"]] == ["root-a", "root-b"]
    assert result["pairs"][0]["adaptive_minus_fixed"]["action_backward_calls"] == 8
    assert result["pairs"][1]["adaptive_minus_fixed"]["action_backward_calls"] == 0
    assert result["p1_dependency"] == "not_passed"
    for flag in ("p3_admitted", "compute_advantage", "walltime_advantage", "efficacy_claim"):
        assert result[flag] is False
    assert result["root_denominator"] == 2
    assert result["all_pairs_within_budget"]
    assert not result["walltime_measured_for_all"]


def test_failure_cost_denominator_overhead_and_horizon_max_not_sum():
    protocol, fixed, adaptive = cohort()
    result = compare_paired(protocol, fixed, adaptive)["pairs"][0]["candidate"]
    assert result["successes"] == 1 and result["planned_branch_denominator"] == 2
    assert result["branches"][1]["status"] == "failed"
    observed = result["resources"]
    assert observed["predict_samples"] == 7
    assert observed["action_backward_calls"] == 2 and observed["input_backward_calls"] == 1
    assert observed["physical_env_steps"] == 3 and observed["physical_branches"] == 2
    assert observed["max_planning_horizon_tokens"] == 3
    assert result["prediction_audit"] == {
        "attempted_predict_samples": 7, "failed_predict_calls": 1,
        "blocked_calls": 1, "incomplete_phases": 1}


@pytest.mark.parametrize("side", [1, 2])
@pytest.mark.parametrize("kind", ["missing", "duplicate", "extra", "both_missing"])
def test_root_census_rejects_missing_duplicate_extra_even_both_sides_drop(side, kind):
    args = list(cohort())
    rows = args[side]
    if kind in ("missing", "both_missing"):
        rows.pop()
        if kind == "both_missing":
            args[3 - side].pop()
    elif kind == "duplicate":
        rows.append(rows[0])
    else:
        rows[0] = replace(rows[0], binding=replace(rows[0].binding, root_id="new-root"))
    with pytest.raises(ValueError):
        compare_paired(*args)


@pytest.mark.parametrize("field", ["protocol_id", "task_id", "frozen_model_id"])
def test_context_mismatch(field):
    protocol, fixed, adaptive = cohort()
    adaptive[0] = replace(adaptive[0], binding=replace(adaptive[0].binding, **{field: "other"}))
    with pytest.raises(ValueError, match="binding"):
        compare_paired(protocol, fixed, adaptive)


@pytest.mark.parametrize("status", ["failed", "blocked", "aborted"])
def test_terminal_failures_are_retained_but_omitting_them_is_rejected(status):
    protocol, fixed, adaptive = cohort()
    branches = (adaptive[0].branches[0], replace(adaptive[0].branches[1], status=status))
    adaptive[0] = replace(adaptive[0], branches=branches)
    assert compare_paired(protocol, fixed, adaptive)["pairs"][0]["candidate"]["branches"][1][
        "status"] == status
    adaptive[0] = replace(adaptive[0], branches=branches[:1])
    with pytest.raises(ValueError, match="branches"):
        compare_paired(protocol, fixed, adaptive)


def test_branch_permutation_and_extra_branch():
    protocol, fixed, adaptive = cohort()
    adaptive[0] = replace(adaptive[0], branches=adaptive[0].branches[::-1])
    assert compare_paired(protocol, fixed, adaptive)["pairs"][0]["equal_resources"]
    adaptive[0] = replace(adaptive[0], branches=(*adaptive[0].branches, replace(
        adaptive[0].branches[0], branch_id="extra")))
    with pytest.raises(ValueError, match="branches"):
        compare_paired(protocol, fixed, adaptive)


@pytest.mark.parametrize("axis", [f.name for f in fields(Resources) if f.name not in (
    "predict_samples", "predict_calls")])
def test_independent_resource_caps_no_cross_axis_compensation(axis):
    protocol, fixed, adaptive = cohort()
    adaptive[0] = replace(adaptive[0], overhead=replace(adaptive[0].overhead, **{axis: 101}))
    result = compare_paired(protocol, fixed, adaptive)
    assert not result["all_pairs_within_budget"]
    assert result["pairs"][0]["candidate_vs_budget"][axis] == "over"
    assert result["pairs"][0]["candidate_vs_budget"]["predict_samples"] == "under"


@pytest.mark.parametrize("mutation", ["samples", "calls", "cap", "method", "binding",
                                     "missing_phases", "completed_failure", "negative",
                                     "bool", "missing_count", "successful_only_phases"])
def test_reject_inconsistent_or_success_only_prediction_receipts(mutation):
    protocol, fixed, adaptive = cohort()
    receipt = deepcopy(adaptive[0].prediction_receipt)
    if mutation == "samples":
        receipt["attempted_predict_samples"] = 6
    elif mutation == "calls":
        receipt["phases"][0]["predict_calls"] = 1
    elif mutation == "cap":
        receipt["predict_sample_limit"] = 99
    elif mutation == "method":
        receipt["method"] = "fixed"
    elif mutation == "binding":
        receipt["binding"]["root_id"] = "other"
    elif mutation == "missing_phases":
        del receipt["phases"]
    elif mutation == "completed_failure":
        receipt["phases"][0]["completed"] = True
    elif mutation == "negative":
        receipt["blocked_calls"] = -1
    elif mutation == "bool":
        receipt["blocked_calls"] = True
    elif mutation == "missing_count":
        del receipt["phases"][0]["failed_predict_calls"]
    else:
        receipt["phases"] = receipt["phases"][1:]
        receipt["attempted_predict_samples"] = 2
    adaptive[0] = replace(adaptive[0], prediction_receipt=receipt)
    with pytest.raises(ValueError):
        compare_paired(protocol, fixed, adaptive)


def test_under_budget_does_not_mean_equal_use():
    protocol, fixed, adaptive = cohort()
    adaptive[0] = replace(adaptive[0], overhead=replace(adaptive[0].overhead, encoder_samples=1))
    result = compare_paired(protocol, fixed, adaptive)
    assert result["all_pairs_within_budget"]
    assert not result["pairs"][0]["equal_resources"]
    assert not result["compute_advantage"]


def test_equal_use_over_caps_is_not_compliance():
    protocol, fixed, adaptive = cohort()
    for rows in (fixed, adaptive):
        rows[0] = replace(rows[0], overhead=replace(rows[0].overhead, physical_env_steps=101))
    result = compare_paired(protocol, fixed, adaptive)
    assert result["pairs"][0]["equal_resources"]
    assert not result["all_pairs_within_budget"]


def test_exact_prediction_cap_and_over_cap_receipt():
    protocol, fixed, adaptive = cohort()
    roots = tuple(replace(root, contract=replace(root.contract, limits=replace(
        root.contract.limits, predict_samples=7))) for root in protocol.roots)
    protocol = replace(protocol, roots=roots)
    for row in (*fixed, *adaptive):
        row.prediction_receipt["predict_sample_limit"] = 7
    assert compare_paired(protocol, fixed, adaptive)["all_pairs_within_budget"]
    adaptive[0].prediction_receipt["phases"][0]["predict_samples"] += 1
    adaptive[0].prediction_receipt["attempted_predict_samples"] += 1
    adaptive[0] = replace(adaptive[0], overhead=replace(adaptive[0].overhead, predict_samples=3))
    with pytest.raises(ValueError, match="cap exceeded"):
        compare_paired(protocol, fixed, adaptive)


def test_both_methods_drop_failed_branch_still_rejected():
    protocol, fixed, adaptive = cohort()
    fixed[0] = replace(fixed[0], branches=fixed[0].branches[:1])
    adaptive[0] = replace(adaptive[0], branches=adaptive[0].branches[:1])
    with pytest.raises(ValueError, match="branches"):
        compare_paired(protocol, fixed, adaptive)


@pytest.mark.parametrize("replay_samples", [2, 2, 1, 2])
def test_search_shaped_synthetic_receipt_must_include_evaluation_replay(replay_samples):
    # Shape follows observed search accounting, but these are synthetic counters,
    # not a reconstruction of unknown real encoder/backward/physical resources.
    protocol, fixed, adaptive = cohort()
    total = 80 + replay_samples
    receipt = deepcopy(adaptive[0].prediction_receipt)
    receipt.update(predict_sample_limit=100, attempted_predict_samples=total, blocked_calls=0)
    receipt["phases"] = [
        {"phase": "planning", "method": "adaptive", "predict_calls": 80,
         "predict_samples": 80, "failed_predict_calls": 0, "completed": True},
        {"phase": "evaluation", "method": "adaptive", "predict_calls": replay_samples,
         "predict_samples": replay_samples, "failed_predict_calls": 0, "completed": True},
    ]
    adaptive[0] = replace(adaptive[0], branches=(
        BranchReceipt("branch-a", "completed", False, resources()),
        BranchReceipt("branch-b", "completed", False, resources())),
        overhead=resources(predict_calls=total, predict_samples=total), prediction_receipt=receipt)
    compared = compare_paired(protocol, fixed, adaptive)["pairs"][0]
    assert compared["candidate"]["prediction_audit"]["attempted_predict_samples"] == total
    adaptive[0] = replace(adaptive[0], overhead=resources(predict_calls=80, predict_samples=80))
    with pytest.raises(ValueError, match="totals mismatch"):
        compare_paired(protocol, fixed, adaptive)


def test_no_success_only_or_walltime_claim_even_when_measured():
    protocol, fixed, adaptive = cohort()
    with pytest.raises(ValueError, match="filtered"):
        compare_paired(protocol, fixed, adaptive, comparison_scope="successful_only")
    with pytest.raises(ValueError, match="walltime"):
        compare_paired(protocol, fixed, adaptive, walltime_advantage_claim=True)
    fixed = [replace(r, walltime_seconds=10) for r in fixed]
    adaptive = [replace(r, walltime_seconds=1) for r in adaptive]
    result = compare_paired(protocol, fixed, adaptive)
    assert result["walltime_measured_for_all"] and not result["walltime_advantage"]
    with pytest.raises(ValueError, match="walltime"):
        compare_paired(protocol, fixed, adaptive, walltime_advantage_claim=True)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "1"])
def test_invalid_walltime(value):
    _, fixed, _ = cohort()
    with pytest.raises(ValueError):
        replace(fixed[0], walltime_seconds=value)


def test_missing_resource_not_zero_and_failures_cannot_be_success():
    with pytest.raises(TypeError):
        Resources(predict_samples=0)
    with pytest.raises(ValueError):
        BranchReceipt("b", "failed", True, resources())
    with pytest.raises(ValueError):
        BranchReceipt("b", "pending", False, resources())


def test_frozen_manifest_duplicate_roots_and_branches():
    protocol = plan()
    with pytest.raises(ValueError):
        replace(protocol, roots=(protocol.roots[0], protocol.roots[0]))
    with pytest.raises(ValueError):
        replace(protocol.roots[0], branch_ids=("b", "b"))
    with pytest.raises(ValueError):
        replace(protocol, adaptive_method="fixed")


def test_real_guard_api_with_non_neural_cuda_metadata_fake_only():
    protocol = plan()
    root = protocol.roots[0]

    class Model:
        def predict(self, embedding, action):
            raise RuntimeError("synthetic failed delegation")

    fake = SimpleNamespace(shape=(3, 1, 2), device=SimpleNamespace(type="cuda", index=0))
    model = Model()
    with PredictionBudget(model, binding=root.contract.binding, method="adaptive",
                          sample_limit=100) as guard:
        with guard.phase("planning", method="adaptive", horizon_tokens=1):
            with pytest.raises(RuntimeError, match="synthetic"):
                model.predict(fake, fake)
    candidate = RunReceipt(root.contract.binding, "adaptive", (
        BranchReceipt("branch-a", "failed", False, resources(predict_calls=1, predict_samples=3)),
        BranchReceipt("branch-b", "blocked", False, resources())), resources(), guard.receipt())
    result = compare_paired(replace(protocol, roots=(root,)), [run(root, "fixed")], [candidate])
    assert result["pairs"][0]["candidate"]["resources"]["predict_samples"] == 3
    # Output cannot mutate source receipts.
    result["pairs"][0]["candidate"]["branches"][0]["resources"]["predict_samples"] = 999
    assert candidate.resources().predict_samples == 3
