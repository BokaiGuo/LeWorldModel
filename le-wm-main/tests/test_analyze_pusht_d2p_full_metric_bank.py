from copy import deepcopy

import pytest

from experiments.spectral_reachability import analyze_pusht_d2p_full_metric_bank as audit


def fixture():
    result = {"schema": "pusht_full_metric_bank.v1", "status": "completed",
        "gates": {k: True for k in audit.REQUIRED_GATES}, "actual_predict_samples": 5056,
        "protocol": {"roots": [str(i) for i in range(4)], "outcome_opened": False}, "rows": []}
    for root in range(4):
        for candidate in range(16):
            branches, imagined = {}, {}
            for mode in audit.MODES:
                suffix = {"predicted_terminal_vector": [1.]*192, "predicted_goal_cost": 1.,
                    "nonlinear": {"initial_objective": 1., "zero_objective": 1., "final_objective": 1.,
                        "iterations": [{"before": 1., "after": 1.}]*4}}
                branches[mode] = {"suffix_receipt": deepcopy(suffix), "final_vector": [1.]*192,
                    "final_pose": {"pose_cost": 1.}, "successor_rgb_sha256": str(candidate),
                    "successor_state": [float(candidate)], "observed_prefix_vector": [float(candidate)]*192,
                    "prefix_pose": {"pose_cost": float(candidate)}}
                imagined[mode] = {**suffix, "linear": {"value": 16.-candidate}}
            result["rows"].append({"root_id": root, "root_sha256": str(root), "proposal_index": candidate,
                "goal_vector": [0.]*192, "predicted": {"predicted_prefix_vector": [float(candidate)]*192,
                    "suffixes": imagined}, "branches": branches})
    return result


def test_whole_bank_arithmetic_accepts_permuted_rows_without_counting_as_independent_roots():
    report = fixture()
    report["rows"].reverse()
    summary, _ = audit.summarize(report)
    assert summary["root_count"] == 4 and summary["all_root_candidate_pairs"] == 64
    assert summary["nonzero_prefix_rgb_sign_agreement"] == 60
    assert not summary["p1_admitted"]


@pytest.mark.parametrize("mutation", ["drop", "duplicate", "prefix_drift", "nan", "bad_incumbent", "drop_gate", "predict_count"])
def test_missing_candidates_and_false_solver_success_are_rejected(mutation):
    report = fixture()
    if mutation == "drop": report["rows"].pop()
    if mutation == "duplicate": report["rows"][-1] = deepcopy(report["rows"][0])
    branch = report["rows"][0]["branches"][audit.MODES[-1]]
    if mutation == "prefix_drift": branch["successor_rgb_sha256"] = "other"
    if mutation == "nan": branch["final_vector"][-1] = float("nan")
    if mutation == "bad_incumbent": branch["suffix_receipt"]["nonlinear"]["final_objective"] = 2.
    if mutation == "drop_gate": report["gates"].pop("cuda_only")
    if mutation == "predict_count": report["actual_predict_samples"] -= 1
    with pytest.raises(ValueError): audit.summarize(report)
