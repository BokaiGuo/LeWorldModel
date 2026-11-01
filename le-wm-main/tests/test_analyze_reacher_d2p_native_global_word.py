"""Endpoint ablation evidence-contract fixtures only."""
import copy

import numpy as np
import pytest

from experiments.spectral_reachability.analyze_reacher_d2p_native_global_word import summarize
from experiments.spectral_reachability.audit_reacher_d2p_native_global_word import select_global_word
from experiments.spectral_reachability.reacher_d2p_suffix_multistart import NAMES


def fixture():
    rows, prefixes, suffixes, old_rows, old_prefixes = [], [], [], [], []
    for seed in range(8):
        root = {"seed": seed, "source_episode": seed + 10, "source_start_row": 20}
        scores = [{"proposal_index": i, "immediate_goal_cost": 1. if i == 0 else .5,
            "refined_goal_cost": .1 if i == 12 else .2} for i in range(17)]
        planned = np.zeros((1, 2, 10)).tolist()
        choice = {**root, "decision": select_global_word(scores), "planned_commands": copy.deepcopy(planned)}
        branch = {**choice, "receipt": {"physical_labels_used": False, "selected": "incumbent", "model_costs": dict.fromkeys(NAMES, .3)},
            "prefix_actions_raw": np.zeros((20, 2)).tolist(),
            "suffix_actions_raw": {k: np.zeros((10, 2)).tolist() for k in ("cold", "carry", "native")}}
        methods = {k: {"action_witness_raw": np.zeros((30, 2)).tolist(),
            "physical": {"final_qpos_distance": .1, "final_rgb_cost": .1}} for k in ("cold", "carry", "native")}
        baseline = {"action_witness_raw": np.zeros((30, 2)).tolist(), "physical": {"final_qpos_distance": .2, "final_rgb_cost": .2}}
        rows.append({**choice, "branch": copy.deepcopy(branch), "methods": methods, "native_selector_baseline": copy.deepcopy(baseline), "gates": {"exact": True}})
        prefixes.append(choice)
        suffixes.append(branch)
        old_rows.append({**root, "methods": {"native_dispatch_native": baseline}})
        old_prefixes.append({**root, "score_rows": scores, "word_records": [{"planned_commands": copy.deepcopy(planned)} for _ in range(17)],
            "native_decision": {"selected_program_index": 1}})
    report = {"schema": "reacher_d2p_native_global_word_audit.v1", "status": "completed",
        "gates": dict.fromkeys(("cuda_only", "all8_roots", "weights_unchanged", "all_exact_replays", "reference_unchanged",
            "prefix_seal_unchanged", "suffix_seal_unchanged", "implementation_unchanged"), True),
        "sealed_protocol": {"root_seeds": list(range(8)), "physical_labels_used_for_planning": False,
            "all_suffix_choices_sealed_before_terminal_execution": True, "non_greedy_margin": 1e-6},
        "rows": rows, "aggregate": {"physical_branches": 32, "mean_distance": {k: float(np.mean([.1] * 8)) for k in methods},
            "baseline_mean_distance": float(np.mean([.2] * 8))}, "actual_model_predict_samples": 100}
    return report, {"rows": prefixes[::-1]}, {"rows": suffixes[::-1]}, {"rows": old_rows[::-1]}, {"rows": old_prefixes[::-1]}


def test_root_identity_and_model_only_selection_verified():
    result = summarize(*fixture())
    assert result["changed_choices"] == 8
    assert result["global_vs_native_selector"]["wins"] == 8
    assert result["physical_branches"] == 32
    assert all(r["global_is_non_greedy"] is False for r in result["rows"])


@pytest.mark.parametrize("kind", ["truthy", "missing_gate", "duplicate", "source", "decision", "selection", "action", "nan", "baseline", "mean", "missing_method", "same_choice"])
def test_rejects_corrupted_ablation_evidence(kind):
    report, prefix, suffix, old, old_prefix = fixture()
    row = report["rows"][0]
    if kind == "truthy":
        report["gates"]["cuda_only"] = 1
    elif kind == "missing_gate":
        del report["gates"]["all_exact_replays"]
    elif kind == "duplicate":
        prefix["rows"][0] = copy.deepcopy(prefix["rows"][1])
    elif kind == "source":
        row["source_episode"] += 1
    elif kind == "decision":
        row["decision"]["selected_program_index"] = 1
    elif kind == "selection":
        row["branch"]["receipt"]["selected"] = "same_prefix_cem"
        suffix["rows"][-1].update(copy.deepcopy(row["branch"]))
    elif kind == "action":
        row["methods"]["native"]["action_witness_raw"][20][0] = .5
    elif kind == "nan":
        row["methods"]["native"]["physical"]["final_rgb_cost"] = float("nan")
    elif kind == "baseline":
        row["native_selector_baseline"]["physical"]["final_qpos_distance"] = .3
    elif kind == "mean":
        report["aggregate"]["baseline_mean_distance"] = 99
    elif kind == "missing_method":
        del row["methods"]["carry"]
    else:
        old_prefix["rows"][-1]["native_decision"]["selected_program_index"] = 12
    with pytest.raises(ValueError):
        summarize(report, prefix, suffix, old, old_prefix)
