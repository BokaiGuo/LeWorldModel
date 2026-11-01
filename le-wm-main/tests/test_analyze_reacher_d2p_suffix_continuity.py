import copy
import json

import pytest

from experiments.spectral_reachability.analyze_reacher_d2p_suffix_continuity import signed_counts, paired_difference, summarize


def fixture():
    rows = []
    for seed in (1, 2):
        cells = [{"proposal_index": i, "imagined_terminal_cost": .1 * i,
            "model_costs": {"cold": .1 * i, "carry": .1 * i + .1, "warm": .1 * i - .1},
            "incumbent": "warm", "outcomes": {name: {"final_rgb_cost": .1 * i, "final_qpos_distance": float(seed)}
                for name in ("cold", "carry", "warm")}, "gates": {"valid": True}} for i in range(17)]
        rows.append({"seed": seed, "decision": {"selected_program_index": 3}, "cells": cells, "selected": copy.deepcopy(cells[3])})
    return {"schema": "reacher_d2p_suffix_continuity_audit.v1", "status": "completed", "gates": {"valid": True},
        "sealed_protocol": {"root_seeds": [1, 2]}, "rows": rows,
        "aggregate": {"mean_selected_distance": {name: 1.5 for name in ("cold", "carry", "warm")}, "incumbent_mean_selected_distance": 1.5}}


def test_margin_counts_exclude_neutral_and_keep_ties():
    assert signed_counts([0, 1, -1, 0], [0, 1, 1, 2]) == {"agreements": 1, "reversals": 1, "ties": 1, "nonzero_candidates": 3}
    json.dumps(signed_counts([0, 1], [0, 1]))


def test_uncertainty_is_root_paired_and_json_serializable():
    result = paired_difference([1, 1, 1], [2, 2, 2])
    assert result["n_roots"] == 3
    assert result["ci95_descriptive"] == [-1., -1.]
    assert result["wins"] == 3
    json.dumps(result)


def test_summary_retains_all_candidate_cells_and_root_uncertainty():
    report = fixture()
    report["rows"].reverse()
    summary = summarize(report)
    assert summary["candidates"] == 34
    assert summary["paired_vs_cold"]["warm"]["n_roots"] == 2
    assert summary["margin_counts"]["state_transfer_same_tail"]["nonzero_candidates"] == 32
    json.dumps(summary)


@pytest.mark.parametrize("kind", ["alias", "duplicate_cell", "incumbent", "mean", "empty_gate", "duplicate_root"])
def test_summary_rejects_changed_evidence(kind):
    report = fixture()
    if kind == "alias":
        report["rows"][0]["selected"]["incumbent"] = "cold"
    elif kind == "duplicate_cell":
        report["rows"][0]["cells"][-1]["proposal_index"] = 15
    elif kind == "incumbent":
        report["rows"][0]["cells"][0]["incumbent"] = "cold"
    elif kind == "mean":
        report["aggregate"]["mean_selected_distance"]["cold"] = .1
    elif kind == "empty_gate":
        report["gates"] = {}
    else:
        report["rows"][1]["seed"] = 1
    with pytest.raises(ValueError):
        summarize(report)
