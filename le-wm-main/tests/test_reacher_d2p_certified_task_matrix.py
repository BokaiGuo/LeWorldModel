import copy

import numpy as np
import pytest

from experiments.spectral_reachability import reacher_d2p_certified_geometry_dispatch as compiler
from experiments.spectral_reachability.audit_reacher_d2p_certified_task_matrix import audit, TASKS


def reports():
    rows = [{"proposal_index": 0, "immediate_goal_cost": 1., "refined_goal_cost": .5},
        {"proposal_index": 1, "immediate_goal_cost": .2, "refined_goal_cost": .3},
        {"proposal_index": 2, "immediate_goal_cost": 1.2, "refined_goal_cost": .1}]
    receipt = {"proposal_rows": rows, "non_greedy_margin": 1e-6,
        "dispatch": compiler.base.select_refined_bank(rows)}
    metrics = {index: {"bounded_quadratic_gain": gain, "bounded_quadratic_gain_upper_bound": gain,
        "objective_certificate_passed": True, "qp_certificate": {"objective_duality_gap_upper_bound": 0},
        "baseline_squared_residual": 1} for index, gain in enumerate([.3, .4, .5])}
    receipt["dispatch"] = compiler.qp.qualify_certified_decision(receipt, metrics)
    receipt["dispatch"]["policy"] = compiler.POLICY
    receipt["common_task_response"] = {"candidate_metrics": {str(key): value for key, value in metrics.items()}}
    base = {"status": "completed", "gates": {"passed": True}, "sealed_protocol": {"compiler_variant": "geometry_certified",
        "root_seeds": [1, 2], "source_roots": [1, 2], "weights_sha256": "model", "adapter_sha256": "stats",
        "source_sha256": "source", "target_delta": [.65, -.65], "implementation_sha256": {"matched": "implementation"}, "bootstrap_seed": 46}}
    result = {}
    for task, limits in TASKS.items():
        value = copy.deepcopy(base)
        value["sealed_protocol"].update({"task": task, "raw_actuator_limits": list(limits)})
        value["rows"] = [{"seed": seed, "source_episode": seed, "source_start_row": seed * 200,
            "task": task, "gates": {"checked": True}, "fd_preflight": {"median_relative_error": .01},
            "compilation": copy.deepcopy(receipt), "mechanism": {"actual_rgb_non_greedy_vs_zero": True,
                "actual_qpos_non_greedy_vs_zero": True},
            "methods": {name: {"action_witness_raw": np.zeros((15, 2)).tolist(), "physical": {"final_qpos_distance": distance},
                "execution_identity": {"initial": {"root": seed, "goal": "same"}},
                "actual_compute": {"model_predict_sample_evaluations": 100, "logical_backward_vjp_rows": 20}}
                for name, distance in (("dispatch", .1), ("same_bank_greedy", .2), ("zero", .3), ("cem_mpc", .1))}} for seed in (1, 2)]
        result[task] = value
    return result


def test_root_identity_join_allows_task_row_reordering():
    value = reports()
    value["weak_shoulder"]["rows"].reverse()
    result = audit(value)
    assert all(cell["model_witnesses"] == 2 for cell in result.values())


def test_task_omission_cannot_select_favorable_condition():
    value = reports()
    del value["weak_elbow"]
    with pytest.raises(ValueError, match="all three"):
        audit(value)


def test_duplicate_root_rejected():
    value = reports()
    value["weak_shoulder"]["rows"].append(copy.deepcopy(value["weak_shoulder"]["rows"][0]))
    with pytest.raises(ValueError, match="identity"):
        audit(value)


def test_physical_identity_mismatch_rejected():
    value = reports()
    value["weak_elbow"]["rows"][0]["methods"]["dispatch"]["execution_identity"]["initial"]["root"] = 12
    with pytest.raises(ValueError, match="physical root"):
        audit(value)


def test_limit_violation_rejected():
    value = reports()
    value["weak_elbow"]["rows"][0]["methods"]["dispatch"]["action_witness_raw"][0][1] = .5
    with pytest.raises(ValueError, match="limits"):
        audit(value)
