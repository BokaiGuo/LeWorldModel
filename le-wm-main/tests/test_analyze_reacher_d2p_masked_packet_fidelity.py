"""Pure arithmetic and tamper fixtures, never neural or efficacy evidence."""
import copy

import numpy as np
import pytest

from experiments.spectral_reachability import analyze_reacher_d2p_masked_packet_fidelity as a


def fixture():
    zero = np.zeros(192).tolist()
    packets = [{"packet": j, "vectors": {k: zero.copy() for k in (*a.KINDS, "goal")},
        "costs": {k: 0. for k in a.KINDS}, "released_vector": zero.copy(),
        "physical": {"packet": j, "observed_vector": zero.copy(),
            "action_sha256": a.digest_raw(np.zeros((5, 2)))}} for j in (1, 2, 3)]
    branch = {"bank_index": 0, "raw_actions": np.zeros((15, 2)).tolist(), "packets": packets,
        "all_physical_packets_exact": True, "released_all_packets_exact": True, "original_terminal_exact": True}
    return {"schema": "reacher_d2p_masked_packet_fidelity.v1", "status": "completed",
        "gates": dict.fromkeys(a.GATES, True), "actual_predict_samples": 1728, "physical_branches": 144,
        "sealed_protocol": {"root_seeds": list(range(16))},
        "rows": [{"seed": s, "source_episode": 100+s, "source_start_row": 1000+s,
            "tasks": {t: {f: copy.deepcopy(branch) for f in a.FAMILIES} for t in a.TASKS}} for s in range(16)]}


def test_full_panel_root_identity_order_invariant():
    r = fixture()
    result = a.summarize(r)
    r["rows"].reverse()
    assert a.summarize(r) == result
    assert result["max_identity_abs_error"] == 0
    assert result["p1_admitted"] is False


@pytest.mark.parametrize("mutation", ["root", "duplicate", "task", "family", "packet", "vector", "nan", "goal", "cost", "native", "observed", "action", "mask", "zero", "gate", "samples", "same_index"])
def test_tamper_and_partial_reports_rejected(mutation):
    r = fixture()
    b = r["rows"][0]["tasks"]["shoulder_only"]["common_native"]
    p = b["packets"][1]
    if mutation == "root": r["rows"].pop()
    elif mutation == "duplicate": r["rows"][-1]["seed"] = 0
    elif mutation == "task": r["rows"][0]["tasks"].pop("wrist_only")
    elif mutation == "family": r["rows"][0]["tasks"]["full_drive"].pop("zero_prefix")
    elif mutation == "packet": b["packets"].pop()
    elif mutation == "vector": p["vectors"]["free"].pop()
    elif mutation == "nan": p["vectors"]["free"][0] = float("nan")
    elif mutation == "goal": p["vectors"]["goal"][0] = 1.
    elif mutation == "cost": p["costs"]["observed"] = 1.
    elif mutation == "native": p["released_vector"][0] = 1.
    elif mutation == "observed": p["physical"]["observed_vector"][0] = 1.
    elif mutation == "action": b["raw_actions"][0][0] = .5
    elif mutation == "mask": b["raw_actions"][0][1] = .5
    elif mutation == "zero": r["rows"][0]["tasks"]["full_drive"]["zero_prefix"]["bank_index"] = 1
    elif mutation == "gate": r["gates"]["cuda_only"] = False
    elif mutation == "samples": r["actual_predict_samples"] = 1727
    else: b["packets"][1]["physical"]["extra"] = 1
    with pytest.raises(ValueError): a.summarize(r)


def test_full_coordinate_decomposition_is_not_a_goal_cost_summary_only():
    rng = np.random.default_rng(2)
    vectors = [rng.normal(size=192) for _ in range(5)]
    result = a.decompose(*vectors)
    assert result["identity_abs_error"] < 1e-12
    assert abs(result["margin_error"] - result["baseline_bias"] - result["response_linear"] - result["response_quadratic"]) < 1e-12
