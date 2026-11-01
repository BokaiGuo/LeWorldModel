"""Source-linked task-construction fixtures, not efficacy evidence."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.spectral_reachability.analyze_reacher_d2p_goal_joint_limits import summarize
from experiments.spectral_reachability.reacher_d2p_feasible_goal import feasible_goal_delta


def fixture():
    seeds, rows, choices, source, originals = list(range(16)), [], [], {}, []
    ranges, means = [[0., 0.], np.deg2rad([-160., 160.]).tolist()], {"legacy": [], "feasible": []}
    for seed in seeds:
        root = np.array([0., -2.6 if seed in (6, 9) else -1.])
        source[seed] = root
        _, metadata = feasible_goal_delta(root, [.65, -.65], [False, True], ranges)
        changed = bool(metadata["flipped_joint_indices"])
        methods = {}
        for family in ("legacy", "feasible"):
            value = .02 if family == "feasible" and changed else 1. if changed else .01
            method = {"action_witness_raw": [[0., 0.]] * 30,
                "physical": {"final_qpos_distance": value, "final_rgb_cost": value}}
            methods[family] = method
            means[family].append(value)
            choices.append({"seed": seed, "family": family, "source_episode": seed, "source_start_row": 100 * seed,
                "admission": deepcopy(metadata), "target_qpos": metadata["historical_target_qpos"] if family == "legacy" else metadata["feasible_target_qpos"],
                "root_rgb_sha256": "root", "goal_rgb_sha256": "goal", "history_rgb_sha256": ["history"],
                "params": np.zeros((1, 6, 10)).tolist(), "action_witness_raw": deepcopy(method["action_witness_raw"]),
                "receipt": {"selection_uses_physical_labels": False, "proposal_trajectory_modified": False,
                    "forward_rollout_evaluations": 384, "trace": [0., 0., 0.]}})
        rows.append({"seed": seed, "source_episode": seed, "source_start_row": seed * 100,
            "changed_task": changed, "methods": methods, "gates": {"replay": True}})
        originals.append({"seed": seed, "source_episode": seed, "source_start_row": seed * 100,
            "methods": {"cem_carry": deepcopy(methods["legacy"])}})
    report = {"schema": "reacher_d2p_goal_joint_limits_audit.v1", "status": "completed",
        "gates": {k: True for k in ("cuda_only", "all16_roots", "all_exact_replays", "weights_unchanged", "selection_seal_unchanged", "references_unchanged", "implementation_unchanged")},
        "sealed_protocol": {"root_seeds": seeds, "physical_labels_used_for_planning": False,
            "all32_plans_sealed_before_terminal_evaluation": True, "target_delta": [.65, -.65],
            "cem": {"horizon": 6, "samples": 128, "steps": 3, "elite": 16, "seed": "root_seed"},
            "scope": "Synthetic contract only, not efficacy."},
        "rows": rows, "actual_model_predict_samples": 73728,
        "aggregate": {"changed_task_roots": [6, 9], "physical_branches": 32,
            "mean_distance": {k: float(np.mean(v)) for k, v in means.items()}}}
    return report, {"rows": choices}, source, originals


def test_preserves_all16_original_roots_and_14_exact_tasks():
    report, selection, source, refs = fixture()
    report["rows"].reverse()
    selection["rows"].reverse()
    refs.reverse()
    summary = summarize(report, selection, source, refs)
    assert summary["changed_task_roots"] == [6, 9]
    assert summary["unchanged_task_roots"] == 14


@pytest.mark.parametrize("error", ["gate", "classification", "source", "duplicate", "missing", "new_target", "changed_valid_image", "changed_valid_physical", "action", "privileged", "compute", "periodic_wrist"])
def test_task_audit_rejects_corruptions(error):
    report, selection, source, refs = fixture()
    if error == "gate":
        report["gates"]["cuda_only"] = 1
    elif error == "classification":
        report["rows"][6]["changed_task"] = False
    elif error == "source":
        source[6][1] += .1
    elif error == "duplicate":
        selection["rows"].append(deepcopy(selection["rows"][0]))
    elif error == "missing":
        report["rows"].pop()
    elif error == "new_target":
        selection["rows"][13]["target_qpos"][1] = 3.
    elif error == "changed_valid_image":
        selection["rows"][1]["goal_rgb_sha256"] = "wrong"
    elif error == "changed_valid_physical":
        report["rows"][0]["methods"]["feasible"]["physical"]["final_rgb_cost"] = .9
    elif error == "action":
        selection["rows"][13]["action_witness_raw"][0][0] = .5
    elif error == "privileged":
        selection["rows"][0]["receipt"]["selection_uses_physical_labels"] = True
    elif error == "compute":
        report["actual_model_predict_samples"] -= 1
    elif error == "periodic_wrist":
        for i in (12, 13):
            selection["rows"][i]["admission"]["limited_joints"][1] = False
    with pytest.raises(ValueError):
        summarize(report, selection, source, refs)
