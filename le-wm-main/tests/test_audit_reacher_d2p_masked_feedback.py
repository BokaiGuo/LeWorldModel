"""Causal extraction/frontier fixtures, not neural or physical evidence."""
import numpy as np
import pytest

from experiments.spectral_reachability import audit_reacher_d2p_masked_feedback as a
from tests.test_analyze_reacher_d2p_masked_packet_fidelity import fixture


def test_whitelist_excludes_all_observed_physical_future_data():
    report = fixture()
    plans = a.initial_plans(report)
    for r in report["rows"]:
        for branches in r["tasks"].values():
            for b in branches.values():
                for p in b["packets"]:
                    p["vectors"]["observed"] = [999.]*192
                    p["vectors"]["actual_history"] = [888.]*192
                    p["physical"] = {"future": "FORBIDDEN"}
    assert a.initial_plans(report) == plans
    assert set(plans[0]["tasks"]["full_drive"]["common_native"]) == {"bank_index", "raw_actions", "waypoints", "goal"}


def stage():
    return {"mode": "goal_feedback", "task": "wrist_only", "family": "common_native", "packet": 2,
        "raw_steps_observed_before_selection": 5, "unexecuted_observations_used": False,
        "raw_actions": np.zeros((5, 2)).tolist(), "target_vector": [0.]*192,
        "predicted_vector": [0.]*192, "optimizer_receipt": {"iterations": 4, "physical_labels_used": False}}


def test_valid_fixed_budget_observed_history_stage():
    a.validate_stage(stage())


@pytest.mark.parametrize("mutation", ["frontier", "future", "mask", "bound", "target", "nan", "iterations", "labels", "first", "replay"])
def test_noncausal_or_unmatched_stages_rejected(mutation):
    s = stage()
    if mutation == "frontier": s["raw_steps_observed_before_selection"] = 10
    elif mutation == "future": s["unexecuted_observations_used"] = True
    elif mutation == "mask": s["raw_actions"][0][0] = .1
    elif mutation == "bound": s["raw_actions"][0][1] = 1.1
    elif mutation == "target": s["target_vector"].pop()
    elif mutation == "nan": s["predicted_vector"][0] = float("nan")
    elif mutation == "iterations": s["optimizer_receipt"]["iterations"] = 3
    elif mutation == "labels": s["optimizer_receipt"]["physical_labels_used"] = True
    elif mutation == "first": s.update(packet=1, raw_steps_observed_before_selection=0)
    else: s["mode"] = "replay"
    with pytest.raises(ValueError): a.validate_stage(s)
