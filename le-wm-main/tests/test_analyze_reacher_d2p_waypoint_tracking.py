"""Complete all-root stored-data contracts, not CPU neural experiments."""
import copy
import hashlib

import numpy as np
import pytest

from experiments.spectral_reachability.analyze_reacher_d2p_waypoint_tracking import FAMILIES, MODES, summarize


def fixture():
    gates = {k: True for k in ("cuda_only", "all8_roots", "all64_branches_and384_packets", "initial_plans_and_waypoints_unchanged",
        "reference_and_decisions_unchanged", "parameters_unchanged", "buffers_unchanged", "all_modules_eval",
        "predict_sample_accounting", "weights_file_unchanged", "implementation_unchanged")}
    protocol = {"root_seeds": list(range(8)), "families": list(FAMILIES), "modes": list(MODES), "packets_per_branch": 6,
        "raw_steps_per_packet": 5, "damping": .01, "iterations": 4, "physical_labels_used_for_planning": False,
        "all_initial_plans_and_waypoints_sealed_before_any_execution": True, "first_packet_unchanged": True,
        "goal_contract": "joint-feasible-v1", "scope": "fixture, not efficacy"}
    report = {"schema": "reacher_d2p_waypoint_tracking.v1", "status": "completed", "gates": gates, "sealed_protocol": protocol,
        "initial_selection_sha256": "initial", "actual_model_predict_samples": 1376, "expected_model_predict_samples": 1376, "rows": []}
    initial, reference, stages = {"rows": []}, {"rows": []}, {}
    actions = np.zeros((5, 2), np.float32)
    sha = hashlib.sha256(actions.tobytes()).hexdigest()
    physical = {"final_qpos_distance": .25, "final_rgb_cost": 1.}
    for seed in range(8):
        identity = {"seed": seed, "source_episode": seed + 100, "source_start_row": seed * 201}
        original = {**identity, "methods": {f + "_carry": {"action_witness_raw": np.zeros((30, 2)).tolist(), "physical": physical.copy()} for f in FAMILIES}}
        plan = {**identity, "goal_vector": [0.] * 192, "plans": {f: {"commands": np.zeros((1, 6, 10)).tolist(),
            "waypoints": [[.5] * 192 for _ in range(6)], "raw_actions": np.zeros((30, 2)).tolist(),
            "baseline_actions_bit_exact": True, "predicted_goal_costs": [.25] * 6} for f in FAMILIES}}
        row = {**identity, "methods": {}}
        for family in FAMILIES:
            for mode in MODES:
                packets = []
                for i in range(6):
                    receipt = {"physical_labels_used": False, "damping": .01, "iterations_requested": 4, "cost_nonincreasing": True,
                        "trace": [{"cost_before": 1., "cost_after": .5}], "iterations_executed": 1, "model_predict_sample_evaluations": 2} if mode == "tracking" and i else None
                    stages[(seed, family, mode, i + 1)] = {**identity, "family": family, "mode": mode, "packet": i + 1,
                        "initial_selection_sha256": "initial", "raw_steps_observed_before_selection": i * 5,
                        "unexecuted_branch_observations_used": False, "target_vector": [.5] * 192, "predicted_vector": [.5] * 192,
                        "raw_actions": actions.tolist(), "receipt": receipt}
                    packets.append({"packet": i + 1, "observed_vector": [1.] * 192, "action_sha256": sha,
                        "root_rgb_sha256": "root", "history_rgb_sha256": ["history"] * 3, "observed_goal_cost": 1.,
                        "observed_waypoint_error_l2": 48 ** .5, "predicted_waypoint_error_l2": 0.})
                row["methods"][family + "_" + mode] = {"packets": packets, "physical": physical.copy(),
                    "action_witness_raw": np.zeros((30, 2)).tolist(), "baseline_carry_metrics_replayed": mode == "replay"}
        report["rows"].append(row)
        initial["rows"].append(plan)
        reference["rows"].append(original)
    return report, initial, reference, stages


def test_complete_independent_accounting_and_identity_join():
    args = fixture()
    result = summarize(*args)
    assert result["actual_model_predict_samples"] == 1376
    assert result["mean_distance"]["native_dispatch_tracking"] == .25
    args[0]["rows"].reverse()
    args[1]["rows"].reverse()
    args[2]["rows"].reverse()
    assert summarize(*args) == result


@pytest.mark.parametrize("kind", ["gate_missing", "gate_nonbool", "gate_false", "root_duplicate", "root_missing", "episode", "count", "label", "first_policy", "missing_stage", "stage_id", "future", "other_branch", "waypoint", "raw_sha", "vectors", "cost", "error", "first_observation", "receipt_label", "receipt_descent", "receipt_budget", "witness", "baseline", "physical", "goal_nan", "initial_plan"])
def test_fail_closed_on_tampering(kind):
    r, initial, old, stages = copy.deepcopy(fixture())
    p = r["rows"][0]["methods"]["native_dispatch_tracking"]["packets"][1]
    stage = stages[(0, "native_dispatch", "tracking", 2)]
    if kind == "gate_missing":
        r["gates"].pop("buffers_unchanged")
    elif kind == "gate_nonbool":
        r["gates"]["cuda_only"] = 1
    elif kind == "gate_false":
        r["gates"]["cuda_only"] = False
    elif kind == "root_duplicate":
        r["rows"][-1]["seed"] = 0
    elif kind == "root_missing":
        r["rows"].pop()
    elif kind == "episode":
        r["rows"][0]["source_episode"] = 999
    elif kind == "count":
        r["actual_model_predict_samples"] += 1
    elif kind == "label":
        r["sealed_protocol"]["physical_labels_used_for_planning"] = True
    elif kind == "first_policy":
        r["sealed_protocol"]["first_packet_unchanged"] = False
    elif kind == "missing_stage":
        stages.pop((0, "zero", "replay", 1))
    elif kind == "stage_id":
        stage["seed"] = 1
    elif kind == "future":
        stage["raw_steps_observed_before_selection"] = 10
    elif kind == "other_branch":
        stage["unexecuted_branch_observations_used"] = True
    elif kind == "waypoint":
        stage["target_vector"][0] += 1e-12
    elif kind == "raw_sha":
        stage["raw_actions"][0][0] = .001
    elif kind == "vectors":
        p["observed_vector"].pop()
    elif kind == "cost":
        p["observed_goal_cost"] = 0.
    elif kind == "error":
        p["observed_waypoint_error_l2"] = 0.
    elif kind == "first_observation":
        r["rows"][0]["methods"]["native_dispatch_tracking"]["packets"][0]["root_rgb_sha256"] = "changed"
    elif kind == "receipt_label":
        stage["receipt"]["physical_labels_used"] = True
    elif kind == "receipt_descent":
        stage["receipt"]["trace"][0]["cost_after"] = 2.
    elif kind == "receipt_budget":
        stage["receipt"]["iterations_requested"] = 32
    elif kind == "witness":
        r["rows"][0]["methods"]["native_dispatch_tracking"]["action_witness_raw"][5][0] = .1
    elif kind == "baseline":
        old["rows"][0]["methods"]["native_dispatch_carry"]["physical"]["final_qpos_distance"] = 1.
    elif kind == "physical":
        r["rows"][0]["methods"]["native_dispatch_tracking"]["physical"]["final_qpos_distance"] = float("nan")
    elif kind == "goal_nan":
        initial["rows"][0]["goal_vector"][0] = float("nan")
    else:
        initial["rows"][0]["plans"]["native_dispatch"]["raw_actions"][0][0] = .1
    with pytest.raises(ValueError):
        summarize(r, initial, old, stages)
