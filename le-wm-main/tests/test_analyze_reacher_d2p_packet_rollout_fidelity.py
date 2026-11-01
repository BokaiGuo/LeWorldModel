"""Independent full-vector arithmetic tests, not neural evidence."""
import copy

import pytest

from experiments.spectral_reachability.analyze_reacher_d2p_packet_rollout_fidelity import summarize


def fixture():
    gates = {k: True for k in ("cuda_only", "all8_roots", "all16_fixed_prefixes_and64_boundaries_exact", "parameters_unchanged",
        "buffers_unchanged", "all_modules_eval", "predict_sample_accounting", "reference_and_decisions_unchanged",
        "weights_file_unchanged", "implementation_unchanged")}
    branch_gates = {k: True for k in ("manual_rollout_terminal_bit_exact", "released_rollout_all4_equivalent", "original_model_cost_replay",
        "original_actual_cost_replay", "all_boundaries_and_actions_exact")}
    branch = {"gates": branch_gates, "released_rollout_max_abs_difference": 0., "released_rollout_vectors": [[1.] * 192 for _ in range(4)],
        "packets": [{"packet": i, "vectors": {"free": [1.] * 192, "teacher_forced": [1.] * 192, "observed": [0.] * 192, "goal": [2.] * 192},
            "costs": {"free": 1., "teacher_forced": 1., "observed": 4.}} for i in range(1, 5)]}
    return {"schema": "reacher_d2p_packet_rollout_fidelity.v1", "status": "completed", "gates": gates, "actual_model_predict_samples": 256,
        "sealed_protocol": {"root_seeds": list(range(8)), "physical_labels_used_for_planning": False, "released_rollout_equivalence": True,
            "native_rollout_atol": 2e-6, "packets_per_branch": 4, "raw_steps_per_packet": 5, "families": ["native_dispatch", "zero"], "scope": "fixture"},
        "rows": [{"seed": i, "native_witness": True, "branches": {f: copy.deepcopy(branch) for f in ("native_dispatch", "zero")}} for i in range(8)]}


def test_complete_arithmetic_and_order_invariance():
    report = fixture()
    expected = summarize(report)
    assert expected["reversed_final_roots"] == {"free": [], "teacher_forced": []}
    assert expected["mean_latent_l2_by_packet"]["free"] == pytest.approx([192 ** .5] * 4)
    report["rows"].reverse()
    assert summarize(report) == expected


@pytest.mark.parametrize("kind", ["partial_gates", "nonbool_gate", "false_gate", "missing_root", "duplicate", "wrong_seed", "count", "labels", "native_disabled", "tolerance_nan", "tolerance_relaxed", "packets", "families", "branch_gate", "witness", "packet_order", "partial_vectors", "nan", "shape", "cost", "first_history", "goal", "native_vectors", "native_summary", "branch_goal"])
def test_tampering_rejected(kind):
    r = fixture()
    b = r["rows"][0]["branches"]["native_dispatch"]
    p = b["packets"][1]
    if kind == "partial_gates":
        r["gates"].pop("buffers_unchanged")
    elif kind == "nonbool_gate":
        r["gates"]["cuda_only"] = 1
    elif kind == "false_gate":
        r["gates"]["cuda_only"] = False
    elif kind == "missing_root":
        r["rows"].pop()
    elif kind == "duplicate":
        r["rows"][-1]["seed"] = 0
    elif kind == "wrong_seed":
        r["rows"][-1]["seed"] = 9
    elif kind == "count":
        r["actual_model_predict_samples"] = 192
    elif kind == "labels":
        r["sealed_protocol"]["physical_labels_used_for_planning"] = True
    elif kind == "native_disabled":
        r["sealed_protocol"]["released_rollout_equivalence"] = False
    elif kind.startswith("tolerance"):
        r["sealed_protocol"]["native_rollout_atol"] = float("nan") if kind.endswith("nan") else 1.
    elif kind == "packets":
        b["packets"].pop()
    elif kind == "families":
        r["rows"][0]["branches"].pop("zero")
    elif kind == "branch_gate":
        b["gates"].pop("original_actual_cost_replay")
    elif kind == "witness":
        r["rows"][0]["native_witness"] = 1
    elif kind == "packet_order":
        b["packets"].reverse()
    elif kind == "partial_vectors":
        p["vectors"].pop("observed")
    elif kind == "nan":
        p["vectors"]["observed"][0] = float("nan")
    elif kind == "shape":
        p["vectors"]["free"].pop()
    elif kind == "cost":
        p["costs"]["free"] = .9
    elif kind == "first_history":
        b["packets"][0]["vectors"]["teacher_forced"][0] += 1e-12
    elif kind == "goal":
        p["vectors"]["goal"][0] += 1e-12
    elif kind == "native_vectors":
        b["released_rollout_vectors"][0] = [1.1] * 192
    elif kind == "native_summary":
        b["released_rollout_max_abs_difference"] = 1.
    else:
        for packet in b["packets"]:
            packet["vectors"]["goal"] = [3.] * 192
            packet["costs"] = {"free": 4., "teacher_forced": 4., "observed": 9.}
    with pytest.raises(ValueError):
        summarize(r)
