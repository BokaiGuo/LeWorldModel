import copy

import numpy as np
import pytest
import torch

from experiments.spectral_reachability import audit_reacher_pc_p1_fd_suffix_saved_20261102_codexexclusive as audit
from experiments.spectral_reachability import reacher_pc_p1_fd_suffix_search_20261102_codexexclusive as search
from experiments.spectral_reachability import reacher_pc_p1_common_tail_tangent_20261102_codexexclusive as common


def make_row():
    def evaluate(x):
        return x[0], torch.stack([-x[1], -2*x[1], x[1], x[1], x[1]])
    bank, receipt = search.generate(torch.zeros(10, dtype=torch.float64), evaluate)
    tails = common.common_tails(torch.zeros(10))
    receipt["common_suffix_parameters"] = tails.tolist()
    prefix = torch.tanh(bank.float()).numpy()[:, None, :]
    tail_norm = torch.tanh(tails).numpy()
    return {
        "seed": 20263160, "status": "complete", "bank_params": bank[:, None, :].tolist(),
        "compiler_receipt": receipt, "prefix_normalized": prefix.tolist(),
        "prefix_raw": prefix.reshape(9, 5, 2).tolist(),
        "tail_normalized": tail_norm.tolist(), "tail_raw": tail_norm.reshape(5, 10, 2).tolist(),
        "predicted_endpoints": np.zeros((9, 5, 192)).tolist(),
        "recalculated_final_costs": [r["final_costs"][:] for r in receipt["proposal_rows"]],
        "census_nominal_samples": {"prefix": 9, "suffix_token": 90},
    }


@pytest.fixture
def row():
    return make_row()


def check(row):
    return audit.audit_row(row, np.zeros(10, np.float32), np.ones(10, np.float32))


def test_valid_separate_development_and_strict_selection(row):
    result = check(row)
    assert result["development_selected_index"] is not None
    assert result["strict_selected_index"] is None
    assert result["goal_q_independent_endpoint_recomputation"] == "unavailable_goal_latent_not_saved"


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r["compiler_receipt"].__setitem__("strict_selected_index", 2), "strict winner"),
    (lambda r: r["compiler_receipt"].__setitem__("development_selected_index", None), "development winner"),
    (lambda r: r["compiler_receipt"]["proposal_rows"][1].__setitem__("eligible", True), "strict selector"),
    (lambda r: r["compiler_receipt"]["proposal_rows"][1].__setitem__("development_eligible", True), "development selector"),
    (lambda r: r["recalculated_final_costs"][1].__setitem__(0, 999), "recalculated scores"),
    (lambda r: r["compiler_receipt"]["proposal_rows"][1]["per_tail_delta"].__setitem__(0, 999), "delta"),
    (lambda r: r["bank_params"][1][0].__setitem__(3, .1), "FD proposal bank"),
    (lambda r: r["compiler_receipt"]["fd_probes"][1].__setitem__("plus_immediate", 999), "FD proposal bank"),
    (lambda r: r["prefix_raw"][1][0].__setitem__(0, .99), "raw action binding"),
    (lambda r: r["prefix_normalized"][1][0].__setitem__(0, .99), "bounded parameter"),
    (lambda r: r["compiler_receipt"]["common_suffix_parameters"][1][0].__setitem__(0, .2), "fixed common tails"),
    (lambda r: r["compiler_receipt"]["logical_costs"].__setitem__("suffix_token_predict_samples", 145), "budget"),
])
def test_tampering_fails(row, mutation, match):
    mutation(row)
    with pytest.raises(ValueError, match=match):
        check(row)


def binding():
    roots = [{"seed": seed, "history_index": i, "root_rgb_sha256": f"root{i}",
              "goal_rgb_sha256": f"goal{i}"} for i, seed in enumerate(audit.ROOTS)]
    preflight = {"roots": roots, "model_parameter_hash_expected": "parameters",
                 "q_report_sha256": "q", "root_source_sha256": "root",
                 "weights_sha256": "weights", "action_stats_sha256": "stats"}
    plan = {"schema": "pc_fd_suffix_cuda_development_v1", "roots": roots, "parameters_sha256": "parameters", "q_sha256": "q",
            "root_source_sha256": "root", "weights_sha256": "weights", "stats_sha256": "stats",
            "preflight_sha256": "preflight",
            "sources": {"/fixture/" + name: "fake" for name in audit.REQUIRED_SOURCE_NAMES}}
    report = {"rows": [dict(root, status="complete") for root in roots],
              "plan_sha256": "plan", "plan_postcheck": True, "frozen_parameters_unchanged": True,
              "parameter_hash_before": "parameters", "parameter_hash_after": "parameters"}
    return report, plan, preflight


def test_plan_identity_failure_receipts_and_source_requirements():
    report, plan, preflight = binding()
    def run():
        audit.audit_binding(report, plan, preflight, plan_hash="plan",
                            preflight_hash="preflight", check_files=False)
    run()
    report["rows"][1]["root_rgb_sha256"] = "tampered"
    with pytest.raises(ValueError, match="root identity"):
        run()
    report, plan, preflight = binding()
    report["rows"][1] = {"seed": audit.ROOTS[1], "status": "error", "failed_stage": "started",
                         "error_type": "ValueError", "error": "saved failure"}
    run()
    del report["rows"][1]["error"]
    with pytest.raises(ValueError, match="failure receipt"):
        run()
    report, plan, preflight = binding()
    plan["preflight_sha256"] = "other"
    with pytest.raises(ValueError, match="plan/preflight"):
        run()
    report, plan, preflight = binding()
    plan["sources"].pop(next(iter(plan["sources"])))
    with pytest.raises(ValueError, match="required plan sources"):
        run()


def test_incomplete_root_not_treated_as_complete(row):
    row["status"] = "error"
    with pytest.raises(ValueError, match="incomplete"):
        check(row)


def test_float32_inputs_with_double_model_scores_reconstruct():
    def evaluate(x):
        z = x.double()
        return (0.23 + z[0] + .17*z[2],
                torch.stack([.2-z[1], .3-2*z[1]+.2*z[3],
                             .4-3*z[1], .5-4*z[1], .6-5*z[1]]))
    bank, receipt = search.generate(torch.zeros(10, dtype=torch.float32), evaluate)
    reconstructed, valid, best = audit._expected_bank(
        receipt["fd_probes"], np.asarray(receipt["proposal_rows"][0]["final_costs"]))
    np.testing.assert_allclose(bank.numpy(), reconstructed, atol=2e-6, rtol=2e-5)
    assert receipt["direction_valid"] == valid
    assert receipt["baseline_best_tail_index"] == best
