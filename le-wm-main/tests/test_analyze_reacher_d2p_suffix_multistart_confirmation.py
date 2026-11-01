"""Prospective evidence-contract fixtures only, no neural experiment."""
import copy

import numpy as np
import pytest

from experiments.spectral_reachability.analyze_reacher_d2p_suffix_multistart_confirmation import FAMILIES, summarize
from experiments.spectral_reachability.reacher_d2p_suffix_multistart import NAMES


def fixture():
    rows, prefixes, suffixes = [], [], []
    for seed in range(8):
        root = {"seed": seed, "source_episode": seed + 10, "source_start_row": 10}
        methods, branches = {}, {}
        prefixes.append({**root, "receipt": {"dispatch": {"reconfiguration_witness_found": True}}})
        for family in FAMILIES:
            prefix, suffix = np.zeros((20, 2)).tolist(), np.zeros((10, 2)).tolist()
            branch = {**root, "family": family, "prefix_actions_raw": prefix,
                "suffix_actions_raw": {name: copy.deepcopy(suffix) for name in NAMES}, "prefix_rgb_cost": .1,
                "receipt": {"model_costs": dict.fromkeys(NAMES, .3), "physical_labels_used": False, "selected": "incumbent"}}
            suffixes.append(copy.deepcopy(branch))
            branches[family] = branch
            for name in (*NAMES, "model_best"):
                methods[family + "_" + name] = {"physical": {"final_qpos_distance": .1, "final_rgb_cost": .2},
                    "action_witness_raw": copy.deepcopy(prefix + suffix)}
        rows.append({**root, "methods": methods, "branches": branches, "gates": {"exact": True}})
    means = {name: float(np.mean([.1] * 8)) for name in methods}
    report = {"schema": "reacher_d2p_suffix_multistart_confirmation.v1", "status": "completed",
        "gates": dict.fromkeys(("cuda_only", "weights_unchanged", "all_declared_roots", "source_episodes_disjoint", "all_execution_gates",
            "all24_branches_per_root", "prefix_seal_unchanged", "suffix_seal_unchanged", "implementation_unchanged"), True),
        "sealed_protocol": {"root_seeds": list(range(8)), "excluded_source_episodes": [99],
            "physical_labels_used_for_planning": False, "all_suffix_choices_sealed_before_terminal_execution": True},
        "rows": rows, "aggregate": {"physical_branches": 192, "mean_distance": means}, "actual_model_predict_samples": 100}
    return report, {"rows": list(reversed(prefixes))}, {"rows": list(reversed(suffixes))}


def test_complete_fresh_identity_joins_are_order_independent():
    result = summarize(*fixture())
    assert result["model_witnesses"] == 8
    assert result["witness_actual_rgb_non_greedy"] == 0
    assert result["physical_branches"] == 192
    assert result["within_prefix_repairs"]["dispatch"]["ties"] == 8


@pytest.mark.parametrize("kind", ["truthy", "missing_gate", "duplicate", "overlap", "suffix", "prefix", "selection", "nan", "mean", "missing_method"])
def test_rejects_invalid_prospective_evidence(kind):
    report, prefixes, suffixes = fixture()
    if kind == "truthy":
        report["gates"]["cuda_only"] = 1
    elif kind == "missing_gate":
        del report["gates"]["weights_unchanged"]
    elif kind == "duplicate":
        suffixes["rows"][0] = copy.deepcopy(suffixes["rows"][1])
    elif kind == "overlap":
        report["sealed_protocol"]["excluded_source_episodes"].append(10)
    elif kind in ("suffix", "prefix"):
        report["rows"][0]["methods"]["dispatch_box_extra_32"]["action_witness_raw"][25 if kind == "suffix" else 0] = [.5, .5]
    elif kind == "selection":
        report["rows"][0]["branches"]["dispatch"]["receipt"]["selected"] = "same_prefix_cem"
    elif kind == "nan":
        report["rows"][0]["methods"]["dispatch_incumbent"]["physical"]["final_qpos_distance"] = float("nan")
    elif kind == "mean":
        report["aggregate"]["mean_distance"]["dispatch_incumbent"] = 99
    else:
        del report["rows"][0]["methods"]["cem_box_extra_8"]
    with pytest.raises(ValueError):
        summarize(report, prefixes, suffixes)
