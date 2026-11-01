"""All-root evidence validation, no neural inference."""
import copy

import numpy as np
import pytest

from experiments.spectral_reachability.analyze_reacher_d2p_suffix_action_chart import NAMES, summarize


def fixture():
    rows, choices = [], []
    for seed in range(8):
        prefix = np.zeros((20, 2)).tolist()
        suffix = np.zeros((10, 2)).tolist()
        methods = {name: {"physical": {"final_qpos_distance": .1, "final_rgb_cost": .2},
            "action_witness_raw": prefix + suffix} for name in NAMES}
        methods["model_best"] = copy.deepcopy(methods["incumbent"])
        costs = dict.fromkeys(NAMES, .3)
        rows.append({"seed": seed, "source_episode": seed + 20, "source_start_row": 1, "methods": methods,
            "model_costs": costs, "model_best": "incumbent", "gates": {"exact": True}})
        choices.append({"seed": seed, "source_episode": seed + 20, "source_start_row": 1,
            "model_costs": costs, "model_best": "incumbent", "prefix_actions_raw": prefix,
            "suffix_actions_raw": dict.fromkeys(NAMES, suffix), "saturation_fraction_at_same_initial_action": .8,
            "chain_rule_relative_error": 1e-7, "fd_relative_errors": [.01] * 8,
            "receipts": {name: {"iterations_executed": 8} for name in NAMES[1:5]}})
    means = {name: float(np.mean([.1] * 8)) for name in (*NAMES, "model_best")}
    return {"schema": "reacher_d2p_suffix_action_chart_audit.v1", "status": "completed",
        "gates": dict.fromkeys(("cuda_only", "weights_unchanged", "all_roots_retained", "all_execution_gates",
            "selection_unchanged", "reference_unchanged", "implementation_unchanged"), True),
        "sealed_protocol": {"root_seeds": list(range(8)), "physical_labels_used_for_planning": False,
            "prefix_fixed": True, "all_roots_sealed_before_terminal_execution": True},
        "rows": rows, "aggregate": {"mean_distance": means, "physical_branches": 48},
        "actual_model_predict_samples": 100}, {"rows": list(reversed(choices))}


def test_all_roots_join_by_identity_with_tie_incumbent():
    result = summarize(*fixture())
    assert result["root_seeds"] == list(range(8))
    assert all(row["model_best"] == "incumbent" for row in result["routes"])
    assert result["paired_to_incumbent"]["distance"]["box_extra_32"]["mean"] == 0


@pytest.mark.parametrize("kind", ["truthy", "missing", "duplicate", "prefix", "suffix", "outcome_choice", "nan", "mean", "branch"])
def test_rejects_invalid_evidence(kind):
    report, choices = fixture()
    if kind == "truthy":
        report["gates"]["cuda_only"] = 1
    elif kind == "missing":
        del report["gates"]["weights_unchanged"]
    elif kind == "duplicate":
        choices["rows"][0] = copy.deepcopy(choices["rows"][1])
    elif kind in ("prefix", "suffix"):
        method = copy.deepcopy(report["rows"][0]["methods"]["box_extra_32"])
        method["action_witness_raw"][0 if kind == "prefix" else 25] = [.5, .5]
        report["rows"][0]["methods"]["box_extra_32"] = method
    elif kind == "outcome_choice":
        report["rows"][0]["model_best"] = "box_extra_32"
        choices["rows"][-1]["model_best"] = "box_extra_32"
    elif kind == "nan":
        report["rows"][0]["methods"]["box_extra_32"]["physical"]["final_rgb_cost"] = float("nan")
    elif kind == "mean":
        report["aggregate"]["mean_distance"]["box_extra_32"] = 99
    else:
        del report["rows"][0]["methods"]["same_prefix_cem"]
    with pytest.raises(ValueError):
        summarize(report, choices)
