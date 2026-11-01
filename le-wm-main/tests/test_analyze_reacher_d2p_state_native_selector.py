"""All-root numerical-contract evidence fixtures, not neural experiments."""
import copy

import numpy as np
import pytest

from experiments.spectral_reachability.analyze_reacher_d2p_state_native_selector import FAMILIES, summarize
from experiments.spectral_reachability.reacher_d2p_refined_dispatch import select_refined_bank
from experiments.spectral_reachability.reacher_d2p_suffix_multistart import NAMES


def fixture():
    rows, prefixes, suffixes = [], [], []
    commands = np.zeros((1, 2, 10)).tolist()
    receipt = {"model_costs": dict.fromkeys(NAMES, .25), "selected": "incumbent", "physical_labels_used": False}
    for seed in range(8):
        root = {"seed": seed, "source_episode": seed + 10, "source_start_row": 10}
        scores = [{"proposal_index": i, "immediate_goal_cost": 1. if i != 1 else .5,
            "refined_goal_cost": .25} for i in range(17)]
        words = [{"index": i, "original_fine8_cost": .5, "native_receipt": copy.deepcopy(receipt),
            "planned_commands": copy.deepcopy(commands), "partition_max_abs_error": 0.} for i in range(17)]
        old_rows = [{**s, "refined_goal_cost": .5} for s in scores]
        decision = select_refined_bank(scores)
        prefixes.append({**root, "historical_decision": select_refined_bank(old_rows), "native_decision": decision,
            "word_records": words, "score_rows": scores, "gates": {"exact": True}})
        methods, branches = {}, {}
        for family in FAMILIES:
            prefix, suffix = np.zeros((20, 2)).tolist(), np.zeros((10, 2)).tolist()
            variants = ("cold", "carry", "native", "historical") if family == "old_dispatch" else ("cold", "carry", "native")
            branch = {**root, "family": family, "prefix_index": None if family == "cem" else 0 if family == "zero" else 1,
                "receipt": copy.deepcopy(receipt), "prefix_actions_raw": prefix,
                "suffix_actions_raw": {v: copy.deepcopy(suffix) for v in variants},
                "planned_commands": copy.deepcopy(commands), "prefix_rgb_cost": .1}
            branches[family] = branch
            suffixes.append(copy.deepcopy(branch))
            for variant in variants:
                methods[family + "_" + variant] = {"physical": {"final_qpos_distance": .1, "final_rgb_cost": .2},
                    "action_witness_raw": copy.deepcopy(prefix + suffix)}
        rows.append({**root, "methods": methods, "branches": branches, "gates": {"exact": True}})
    report = {"schema": "reacher_d2p_state_native_selector_audit.v1", "status": "completed",
        "gates": dict.fromkeys(("cuda_only", "weights_unchanged", "all_declared_roots", "all17_words_every_root", "all_core_equivalence_gates",
            "all_physical_gates", "prefix_seal_unchanged", "suffix_seal_unchanged", "reference_unchanged", "implementation_unchanged"), True),
        "sealed_protocol": {"root_seeds": list(range(8)), "physical_labels_used_for_planning": False,
            "all_suffix_choices_sealed_before_terminal_execution": True, "non_greedy_margin": 1e-6, "partition_atol": 2e-6},
        "rows": list(reversed(rows)), "aggregate": {"physical_branches": 128,
            "mean_distance": {name: float(np.mean([.1] * 8)) for name in methods}}, "actual_model_predict_samples": 100}
    return report, {"rows": list(reversed(prefixes))}, {"rows": list(reversed(suffixes))}


def test_identity_pairing_and_model_only_reselection():
    result = summarize(*fixture())
    assert result["root_seeds"] == list(range(8))
    assert result["physical_branches"] == 128
    assert result["imagined_words"] == 136
    assert result["native_witnesses"] == 0
    assert result["comparisons"]["greedy_native"]["ties"] == 8
    assert all(r["global_model_best_word_index"] == 0 and r["model_cost_above_global_min"] == 0 for r in result["rows"])


@pytest.mark.parametrize("kind", ["valid", "missing_gate", "missing_receipt", "wrong_delta", "unknown_contract", "legacy_claim"])
def test_explicit_goal_version_cannot_be_relabeled(kind):
    from experiments.spectral_reachability.reacher_d2p_feasible_goal import feasible_goal_delta

    report, prefix, suffix = fixture()
    report["sealed_protocol"].update(goal_contract="joint-feasible-v1", target_delta=[.65, -.65])
    report["gates"].update(explicit_goal_contract=True, all_goal_admissions=True)
    _, admission = feasible_goal_delta([0., -2.6], [.65, -.65], [False, True], [[0, 0], [-2.7925, 2.7925]])
    for row in prefix["rows"]:
        row["goal_admission"] = copy.deepcopy(admission)
    if kind == "missing_gate":
        del report["gates"]["all_goal_admissions"]
    elif kind == "missing_receipt":
        del prefix["rows"][0]["goal_admission"]
    elif kind == "wrong_delta":
        report["sealed_protocol"]["target_delta"] = [.1, .1]
    elif kind == "unknown_contract":
        report["sealed_protocol"]["goal_contract"] = "unknown"
    elif kind == "legacy_claim":
        del report["sealed_protocol"]["goal_contract"]
    if kind == "valid":
        result = summarize(report, prefix, suffix)
        assert result["goal_contract"] == "joint-feasible-v1"
        assert result["goal_reflected_roots"] == list(range(8))
    else:
        with pytest.raises(ValueError):
            summarize(report, prefix, suffix)


@pytest.mark.parametrize("overlap", [False, True])
def test_fresh_confirmation_requires_source_episode_isolation(overlap):
    report, prefix, suffix = fixture()
    report["schema"] = "reacher_d2p_state_native_confirmation.v1"
    del report["gates"]["reference_unchanged"]
    report["gates"].update(dict.fromkeys(("source_episodes_disjoint", "alignment_unchanged", "exclusions_unchanged"), True))
    report["sealed_protocol"]["excluded_source_episodes"] = [10 if overlap else 99]
    if overlap:
        with pytest.raises(ValueError, match="excluded-episode-disjoint"):
            summarize(report, prefix, suffix)
    else:
        assert "Source-episode-disjoint" in summarize(report, prefix, suffix)["scope"]


@pytest.mark.parametrize("kind", ["truthy", "missing_gate", "status", "physical_planning", "unsealed", "duplicate_root",
    "duplicate_prefix", "duplicate_suffix", "source", "missing_word", "extra_score", "score_order", "selection", "old_selection",
    "negative_score", "command_shape", "command_nan", "command_bound", "partition", "suffix_selection", "carry", "action", "nan",
    "aggregate", "branch_count", "missing_method", "fallback", "prefix_nan"])
def test_rejects_broken_numerical_contracts(kind):
    report, prefix, suffix = fixture()
    choice = prefix["rows"][0]
    row = report["rows"][0]
    if kind == "truthy":
        report["gates"]["cuda_only"] = 1
    elif kind == "missing_gate":
        del report["gates"]["weights_unchanged"]
    elif kind == "status":
        report["status"] = "running"
    elif kind == "physical_planning":
        report["sealed_protocol"]["physical_labels_used_for_planning"] = True
    elif kind == "unsealed":
        report["sealed_protocol"]["all_suffix_choices_sealed_before_terminal_execution"] = False
    elif kind == "duplicate_root":
        report["rows"][0] = copy.deepcopy(report["rows"][1])
    elif kind == "duplicate_prefix":
        prefix["rows"][0] = copy.deepcopy(prefix["rows"][1])
    elif kind == "duplicate_suffix":
        suffix["rows"][0] = copy.deepcopy(suffix["rows"][1])
    elif kind == "source":
        choice["source_episode"] += 1
    elif kind == "missing_word":
        choice["word_records"].pop()
    elif kind == "extra_score":
        choice["score_rows"].append(copy.deepcopy(choice["score_rows"][-1]))
    elif kind == "score_order":
        choice["score_rows"][0]["proposal_index"] = 1
    elif kind == "selection":
        choice["native_decision"]["selected_program_index"] = 2
    elif kind == "old_selection":
        choice["historical_decision"]["selected_program_index"] = 2
    elif kind == "negative_score":
        choice["word_records"][0]["original_fine8_cost"] = -1.
    elif kind.startswith("command_"):
        if kind == "command_shape":
            choice["word_records"][0]["planned_commands"] = [[0.]]
        else:
            choice["word_records"][0]["planned_commands"][0][0][0] = float("nan") if kind == "command_nan" else 1.1
    elif kind == "partition":
        choice["word_records"][0]["partition_max_abs_error"] = -.1
    elif kind in ("suffix_selection", "carry", "prefix_nan"):
        branch = row["branches"]["native_dispatch"]
        if kind == "carry":
            branch["planned_commands"][0][0][0] = .5
        elif kind == "suffix_selection":
            branch["receipt"]["selected"] = "same_prefix_cem"
        else:
            branch["prefix_rgb_cost"] = float("nan")
        sealed = next(r for r in suffix["rows"] if (r["seed"], r["family"]) == (row["seed"], "native_dispatch"))
        sealed.update(copy.deepcopy(branch))
    elif kind == "action":
        row["methods"]["zero_native"]["action_witness_raw"][20][0] = .5
    elif kind == "nan":
        row["methods"]["zero_native"]["physical"]["final_rgb_cost"] = float("nan")
    elif kind == "aggregate":
        report["aggregate"]["mean_distance"]["zero_native"] = 99
    elif kind == "branch_count":
        report["aggregate"]["physical_branches"] = 127
    elif kind == "missing_method":
        del row["methods"]["zero_native"]
    else:
        row["methods"]["native_dispatch_native"]["physical"]["final_qpos_distance"] = .2
    with pytest.raises(ValueError):
        summarize(report, prefix, suffix)
