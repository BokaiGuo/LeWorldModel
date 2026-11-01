import copy
import json

import pytest

from experiments.spectral_reachability.analyze_reacher_d2p_suffix_continuation_confirmation import NAMES, summarize


def fixture():
    rows, choices = [], []
    for seed in (1, 2):
        decision = {"selected_program_index": 1, "same_bank_greedy_index": 2, "reconfiguration_witness_found": True}
        methods = {name: {"physical": {"final_qpos_distance": 1.}, "action_witness_raw": [[0., 0.]] * 30,
            "prefix_boundaries": [{"raw_steps_executed": step} for step in (5, 10, 15, 20)]} for name in NAMES}
        branches = {name: {"receipt": {"physical_labels_used": False, "selected": "warm", "model_costs": {"cold": 2., "warm": 1.}}}
            for name in ("dispatch", "greedy", "zero", "flat")}
        rows.append({"seed": seed, "source_episode": seed + 10, "source_start_row": seed * 100, "decision": decision,
            "methods": methods, "branches": branches, "actual_rgb_non_greedy": True, "gates": {"valid": True}})
        choices.append({"seed": seed, "source_episode": seed + 10, "source_start_row": seed * 100, "receipt": {"dispatch": copy.deepcopy(decision)}})
    report = {"schema": "reacher_d2p_suffix_continuation_confirmation.v1", "status": "completed", "gates": {"valid": True},
        "sealed_protocol": {"root_seeds": [1, 2], "excluded_source_episodes": [50]}, "rows": rows,
        "aggregate": {"mean_final_distance": {name: 1. for name in NAMES}}}
    return report, {"rows": choices}


def test_all_methods_and_identity_join_are_retained():
    report, selections = fixture()
    selections["rows"].reverse()
    result = summarize(report, selections)
    assert result["roots"] == 2
    assert result["physical_branches"] == 20
    assert result["within_prefix_repairs"]["dispatch"]["ties"] == 2
    json.dumps(result)


@pytest.mark.parametrize("kind", ["overlap", "duplicate", "missing_method", "prefix", "outcome_selection", "mean"])
def test_prospective_summary_rejects_contract_drift(kind):
    report, selections = fixture()
    if kind == "overlap":
        report["sealed_protocol"]["excluded_source_episodes"].append(11)
    elif kind == "duplicate":
        selections["rows"][1] = copy.deepcopy(selections["rows"][0])
    elif kind == "missing_method":
        report["rows"][0]["methods"].pop("cem_incumbent")
    elif kind == "prefix":
        report["rows"][0]["methods"]["dispatch_incumbent"]["action_witness_raw"][0] = [.1, .1]
    elif kind == "outcome_selection":
        report["rows"][0]["branches"]["dispatch"]["receipt"]["physical_labels_used"] = True
    else:
        report["aggregate"]["mean_final_distance"]["dispatch_incumbent"] = 2.
    with pytest.raises(ValueError):
        summarize(report, selections)
