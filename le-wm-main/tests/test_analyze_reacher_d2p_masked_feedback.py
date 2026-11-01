"""Synthetic whole-panel seal/arithmetic fixtures, not CUDA evidence."""
import copy

import pytest

from experiments.spectral_reachability import analyze_reacher_d2p_masked_feedback as a
from tests.test_analyze_reacher_d2p_masked_packet_fidelity import fixture as packet_fixture


def fixture():
    ref = packet_fixture()
    for row in ref["rows"]:
        for b in row["tasks"].values():
            for f in b.values():
                for p in f["packets"]: p["physical"]["qpos"] = [0., 0.]
    plans = {"rows": a.initial_plans(ref)}
    rows, stages = [], {}
    for row in ref["rows"]:
        seed, tasks = row["seed"], {}
        for task in a.TASKS:
            families = {}
            for family in a.FAMILIES:
                modes = {}
                for mode in a.MODES:
                    packets = []
                    for j in range(3):
                        s = {"seed": seed, "task": task, "family": family, "mode": mode, "packet": j+1,
                            "initial_plans_sha256": "plans", "parent_execution_sha256": "plans" if j == 0 else f"execution{j}",
                            "input_root_rgb_sha256": "root", "raw_steps_observed_before_selection": 5*j,
                            "unexecuted_observations_used": False, "raw_actions": [[0., 0.]]*5,
                            "target_vector": [0.]*192, "predicted_vector": [0.]*192, "optimizer_receipt": None}
                        if mode != "replay" and j > 0:
                            s["optimizer_receipt"] = {"iterations": 4, "physical_labels_used": False,
                                "trace": [{"iteration": i+1, "cost_before": 0., "cost_after": 0., "accepted_scale": 0.,
                                    "qp_certificate": {"active_set_terminated": True, "box_feasibility_violation": 0.,
                                        "device": "cuda:0", "objective_duality_gap_upper_bound": 0.}} for i in range(4)]}
                        physical = copy.deepcopy(row["tasks"][task][family]["packets"][j]["physical"])
                        physical["root_rgb_sha256"] = "root"
                        row["tasks"][task][family]["packets"][j]["physical"] = copy.deepcopy(physical)
                        packets.append({"physical": physical, "observed_goal_cost": 0., "observed_waypoint_l2": 0.,
                            "execution_sha256": f"execution{j+1}"})
                        stages[(seed, task, family, mode, j+1)] = s
                    modes[mode] = {"packets": packets, "raw_actions": [[0., 0.]]*15, "final_rgb_cost": 0., "final_qpos_distance": 0.}
                families[family] = {"bank_index": 0, "modes": modes}
            tasks[task] = families
        rows.append({"seed": seed, "source_episode": row["source_episode"], "source_start_row": row["source_start_row"], "tasks": tasks})
    report = {"schema": "reacher_d2p_masked_feedback.v1", "status": "completed", "gates": dict.fromkeys(a.GATES, True),
        "p1_admitted": False, "actual_predict_samples": 19728, "physical_branches": 432, "plans_sha256": "plans",
        "sealed_protocol": {"families": list(a.FAMILIES), "tasks": list(a.TASKS), "modes": list(a.MODES),
            "iterations": 4, "damping": .01, "prefixes_and_initial_selection_unchanged": True, "root_seeds": list(range(16))}, "rows": rows}
    admissions = {s: {"feasible_target_qpos": [0., 0.], "limited_joints": [False, True]} for s in range(16)}
    return report, plans, ref, stages, admissions


def test_all_complete_root_pairs_invariant_to_measurement_order():
    values = fixture()
    result = a.summarize(*values)
    values[0]["rows"].reverse()
    assert a.summarize(*values) == result
    assert result["p1_admitted"] is False


@pytest.mark.parametrize("mutation", ["root", "duplicate", "stage", "parent", "frontier", "target", "action", "goal_cost", "distance", "optimizer", "decline", "sample", "gate", "admit"])
def test_partial_tampered_and_noncausal_results_rejected(mutation):
    values = fixture()
    r, _, _, stages, _ = values
    stage = stages[(0, "full_drive", "common_native", "goal_feedback", 2)]
    branch = r["rows"][0]["tasks"]["full_drive"]["common_native"]["modes"]["goal_feedback"]
    if mutation == "root": r["rows"].pop()
    elif mutation == "duplicate": r["rows"][-1]["seed"] = 0
    elif mutation == "stage": stages.pop((0, "full_drive", "common_native", "replay", 1))
    elif mutation == "parent": stage["parent_execution_sha256"] = "future"
    elif mutation == "frontier": stage["raw_steps_observed_before_selection"] = 10
    elif mutation == "target": stage["target_vector"][0] = 1.
    elif mutation == "action": stage["raw_actions"] = [[.5, .5]]*5
    elif mutation == "goal_cost": branch["packets"][1]["observed_goal_cost"] = 1.
    elif mutation == "distance": branch["final_qpos_distance"] = 1.
    elif mutation == "optimizer": stage["optimizer_receipt"]["trace"].pop()
    elif mutation == "decline": stage["optimizer_receipt"]["trace"][0]["cost_after"] = 1.
    elif mutation == "sample": r["actual_predict_samples"] -= 1
    elif mutation == "gate": r["gates"]["buffers_unchanged"] = False
    else: r["p1_admitted"] = True
    with pytest.raises(ValueError): a.summarize(*values)
