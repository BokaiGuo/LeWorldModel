from copy import deepcopy

import numpy as np
import pytest

from experiments.spectral_reachability import analyze_pusht_d2p_physical_calibration as audit


def fixtures():
    def branch(value, pose):
        return {"prefix_prediction_before_execution": {"predicted_successor_vector": [value]*192,
                    "goal_vector": [0.]*192, "predicted_goal_cost": value**2, "live_action_roundtrip_max_abs": 0.},
                "suffix_recompile": {"observed_successor_vector": [value]*192, "goal_vector": [0.]*192,
                    "observed_successor_goal_cost": value**2, "value": 1., "projection_sha256": "same"},
                "after_prefix_label": {"pose_cost": pose}, "final_label": {"pose_cost": pose},
                "warmup_rgb_sha256": "history", "goal_rgb_sha256": "goal"}
    report = {"schema": "pusht_d2p_p1_physical_calibration.v3", "status": "fail",
        "gates": {k: True for k in audit.INTEGRITY_GATES | audit.METRIC_GATES},
        "protocol": {"outcome_roots_opened": False}, "rows": []}
    report["gates"]["all_selected_prefixes_physically_no_more_goal_greedy_than_raw_zero"] = False
    upstream = {"rows": []}
    for index in range(2):
        key = str(index)
        report["rows"].append({"root_sha256": key, "root_id": index, "model_eligible": True,
            "selected": branch(2., .5), "raw_zero_control": branch(1., 1.)})
        upstream["rows"].append({"root_sha256": key, "history_rgb_sha256": "history", "goal_rgb_sha256": "goal",
            "compilation": {"model_margin_vs_zero": 3., "selected_proposal_index": 1}})
    return report, upstream


def test_full_coordinates_and_identity_join_separate_the_metrics():
    report, upstream = fixtures()
    upstream["rows"].reverse()
    result = audit.summarize(report, upstream)
    assert result["rgb_non_greedy_count"] == 2
    assert result["pose_non_greedy_count"] == 0
    assert result["rgb_pose_sign_disagreement_count"] == 2
    assert not result["p1_admitted"]


@pytest.mark.parametrize("mutation", ["duplicate", "partial_vector", "changed_coordinate", "dropped_gate", "forged_pose_gate", "outcome"])
def test_incomplete_or_tampered_receipts_rejected(mutation):
    report, upstream = fixtures()
    if mutation == "duplicate": report["rows"][1]["root_sha256"] = "0"
    if mutation == "partial_vector": report["rows"][0]["selected"]["suffix_recompile"]["goal_vector"] = [0.]
    if mutation == "changed_coordinate": report["rows"][0]["selected"]["suffix_recompile"]["observed_successor_vector"][-1] = 10.
    if mutation == "dropped_gate": report["gates"].pop("cuda_only")
    if mutation == "forged_pose_gate":
        report["gates"]["all_selected_prefixes_physically_no_more_goal_greedy_than_raw_zero"] = True
        report["status"] = "pass"
    if mutation == "outcome": report["protocol"]["outcome_roots_opened"] = True
    with pytest.raises(ValueError): audit.summarize(report, upstream)
