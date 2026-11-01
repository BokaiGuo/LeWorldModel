"""Synthetic seal-contract fixtures only; never treated as efficacy results."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.spectral_reachability.analyze_reacher_d2p_packet_mpc import summarize


def fixture():
    seeds, stages, rows, originals, predict = list(range(8)), [{"rows": []} for _ in range(6)], [], [], 0
    for seed in seeds:
        base = {"action_witness_raw": [[0., 0.]] * 30,
            "physical": {"final_qpos_distance": float(seed), "final_rgb_cost": float(seed)}}
        methods = {"openloop": deepcopy(base)}
        for mode in ("cold", "carry"):
            decisions, boundaries = [], []
            for t in range(6):
                params = np.zeros((1, 6 - t, 10)).tolist()
                carried = params if mode == "carry" and t else None
                costs = {"searched": 0., **({"carried": 0.} if carried is not None else {})}
                count = (384 + len(costs)) * (6 - t)
                predict += count
                decision = {"seed": seed, "mode": mode, "stage": t, "remaining_horizon": 6 - t,
                    "source_episode": seed, "source_start_row": 0,
                    "root_rgb_sha256": f"rgb_{seed}_{t}", "history_rgb_sha256": [f"history_{seed}_{t}"],
                    "searched_params": deepcopy(params), "carried_params": deepcopy(carried), "chosen_params": deepcopy(params),
                    "first_packet_raw": [[0., 0.]] * 5,
                    "selection": {"current_model_costs": costs, "selected": "carried" if carried is not None else "searched",
                        "physical_labels_used": False, "inverse_tanh_used": False, "carry_rescored_on_current_rgb": carried is not None},
                    "search_receipt": {"selection_uses_physical_labels": False, "proposal_trajectory_modified": False,
                        "forward_rollout_evaluations": 384, "trace": [0., 0., 0.]},
                    "actual_model_predict_samples": count}
                decisions.append(decision)
                stages[t]["rows"].append(deepcopy(decision))
                boundaries.append({"root_rgb_sha256": f"rgb_{seed}_{t + 1}", "history_rgb_sha256": [f"history_{seed}_{t + 1}"]})
            methods[mode] = {**deepcopy(base), "stages": decisions, "boundaries": boundaries}
        rows.append({"seed": seed, "source_episode": seed, "source_start_row": 0,
            "methods": methods, "gates": {"exact": True}})
        originals.append({"seed": seed, "source_episode": seed, "source_start_row": 0, "methods": {"cem_carry": deepcopy(base)}})
    report = {"schema": "reacher_d2p_packet_mpc_audit.v1", "status": "completed",
        "gates": {k: True for k in ("cuda_only", "weights_unchanged", "all8_roots", "all_exact_replays", "all_stage_seals_unchanged", "reference_unchanged", "implementation_unchanged")},
        "sealed_protocol": {"root_seeds": seeds, "physical_labels_used_for_planning": False,
            "all_stage_decisions_sealed_before_stage_execution": True,
            "all_final_packet_decisions_sealed_before_any_terminal_evaluation": True,
            "modes": ["cold", "carry"], "initial_horizon": 6, "packet_raw_steps": 5, "physical_raw_steps": 30,
            "cem": {"samples": 128, "steps": 3, "elite": 16}, "scope": "Synthetic contract fixture, not efficacy."},
        "stage_files": [{"stage": t} for t in range(6)], "rows": rows,
        "aggregate": {"mean_distance": {k: 3.5 for k in ("openloop", "cold", "carry")}},
        "actual_model_predict_samples": predict}
    return report, stages, {"rows": originals}


def test_every_root_and_stage_is_identity_joined():
    report, stages, reference = fixture()
    report["rows"].reverse()
    reference["rows"].reverse()
    for stage in stages:
        stage["rows"].reverse()
    summary = summarize(report, stages, reference)
    assert summary["root_seeds"] == list(range(8))
    assert summary["carried_plan_selections"] == 40
    assert summary["unique_policy_branches"] == 24
    assert summary["total_physical_30step_executions"] == 40


@pytest.mark.parametrize("error", ["missing_gate", "true_integer", "duplicate_root", "duplicate_stage", "missing_decision", "physical_label", "old_history", "action", "accounting", "baseline", "carry_shift", "cost_selection"])
def test_invalid_evidence_fails_closed(error):
    report, stages, reference = fixture()
    decision = report["rows"][0]["methods"]["carry"]["stages"][1]
    if error == "missing_gate":
        del report["gates"]["cuda_only"]
    elif error == "true_integer":
        report["gates"]["cuda_only"] = 1
    elif error == "duplicate_root":
        report["rows"][1] = deepcopy(report["rows"][0])
    elif error == "duplicate_stage":
        stages[1]["rows"].append(deepcopy(stages[1]["rows"][0]))
    elif error == "missing_decision":
        stages[0]["rows"].pop()
    elif error == "baseline":
        reference["rows"][0]["methods"]["cem_carry"]["physical"]["final_rgb_cost"] += .1
    else:
        if error == "physical_label":
            decision["selection"]["physical_labels_used"] = True
        elif error == "old_history":
            decision["history_rgb_sha256"] = ["stale"]
        elif error == "action":
            decision["first_packet_raw"][0][0] = .1
        elif error == "accounting":
            decision["actual_model_predict_samples"] += 1
        elif error == "carry_shift":
            decision["carried_params"][0][0][0] = .1
        elif error == "cost_selection":
            decision["selection"]["current_model_costs"]["carried"] = 1.
        index = next(i for i, d in enumerate(stages[1]["rows"]) if d["seed"] == 0 and d["mode"] == "carry")
        stages[1]["rows"][index] = deepcopy(decision)
    with pytest.raises(ValueError):
        summarize(report, stages, reference)
