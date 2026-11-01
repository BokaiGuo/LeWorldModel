"""Coverage evidence fixtures only; not neural or predictive evidence."""
import copy

import numpy as np
import pytest

from experiments.spectral_reachability.analyze_reacher_d2p_cem_budget import summarize


def fixture():
    rows, selected, reference = [], [], []
    names = ("full_original", "full_wide", "fixed_native", "fixed_wide")
    for seed in range(8):
        root = {"seed": seed, "source_episode": seed + 10, "source_start_row": 20}
        actions = {k: np.zeros((30, 2)).tolist() for k in names}
        receipt = {"selection_uses_physical_labels": False, "forward_rollout_evaluations": 12288, "trace": [1.] * 12}
        choice = {**root, "physical_labels_used": False, "full_costs": {"original": .2, "wide": .1},
            "suffix_costs": {"native": .1, "wide": .2}, "full_selected": "wide", "suffix_selected": "native",
            "full_receipt": copy.deepcopy(receipt), "suffix_receipt": copy.deepcopy(receipt),
            "original_receipt": {"selection_uses_physical_labels": False, "forward_rollout_evaluations": 384, "trace": [1.] * 3},
            "action_witnesses_raw": actions}
        methods = {k: {"action_witness_raw": copy.deepcopy(actions[k]), "physical": {"final_qpos_distance": .1, "final_rgb_cost": .1}} for k in names}
        selected.append(copy.deepcopy(choice))
        rows.append({"seed": seed, "choice": choice, "methods": methods, "gates": {"exact": True}})
        reference.append({**root, "methods": {"native_dispatch_native": copy.deepcopy(methods["fixed_native"])}})
    report = {"schema": "reacher_d2p_cem_budget_audit.v1", "status": "completed",
        "gates": dict.fromkeys(("cuda_only", "all8_roots", "weights_unchanged", "all_exact_replays", "reference_unchanged",
            "selection_seal_unchanged", "implementation_unchanged"), True),
        "sealed_protocol": {"root_seeds": list(range(8)), "physical_labels_used_for_planning": False,
            "all_choices_sealed_before_terminal_execution": True, "wide_cem": {"samples": 1024, "steps": 12},
            "original_cem": {"samples": 128, "steps": 3}}, "rows": rows,
        "aggregate": {"physical_branches": 32, "mean_distance": {k: float(np.mean([.1] * 8)) for k in (*names, "full_model_best", "suffix_model_best")}},
        "actual_model_predict_samples": 100}
    return report, {"rows": selected[::-1]}, {"rows": reference[::-1]}


def test_complete_identity_join_and_all_branches_verified():
    result = summarize(*fixture())
    assert result["root_seeds"] == list(range(8))
    assert result["wide_full_vs_original"]["ties"] == 8
    assert result["physical_branches"] == 32


@pytest.mark.parametrize("kind", ["truthy", "duplicate", "decision", "selection", "receipt", "action", "prefix", "baseline", "cost", "mean"])
def test_rejects_bad_coverage_evidence(kind):
    report, selected, reference = fixture()
    row = report["rows"][0]
    if kind == "truthy":
        report["gates"]["cuda_only"] = 1
    elif kind == "duplicate":
        selected["rows"][0] = copy.deepcopy(selected["rows"][1])
    elif kind in ("decision", "receipt", "cost"):
        if kind == "decision":
            row["choice"]["full_selected"] = "original"
        elif kind == "receipt":
            row["choice"]["full_receipt"]["forward_rollout_evaluations"] = 1
        else:
            row["choice"]["full_costs"]["wide"] = float("nan")
        selected["rows"][-1] = copy.deepcopy(row["choice"])
    elif kind == "selection":
        row["choice"]["source_episode"] = 99
    elif kind == "action":
        row["methods"]["full_wide"]["action_witness_raw"][25][0] = .5
    elif kind == "prefix":
        row["methods"]["fixed_wide"]["action_witness_raw"][0][0] = .5
        row["choice"]["action_witnesses_raw"]["fixed_wide"][0][0] = .5
        selected["rows"][-1] = copy.deepcopy(row["choice"])
    elif kind == "baseline":
        reference["rows"][-1]["methods"]["native_dispatch_native"]["physical"]["final_qpos_distance"] = .2
    else:
        report["aggregate"]["mean_distance"]["full_wide"] = 99
    with pytest.raises(ValueError):
        summarize(report, selected, reference)
