from copy import deepcopy

import pytest

from experiments.spectral_reachability import p1_full_cem_cost_compare_20261101_main as m


def fixture(arm):
    episode = "s0_full_cem" if arm == "cem" else "s0_structured"
    charge = {key: 0 for key in m.AXES}
    charge.update(predict_calls=2, predict_samples=2, action_encode_calls=2)
    receipt = {"account": {"status": "closed_unverified", "charged_upper_bounds": charge}}
    if arm == "structured":
        receipt["module_hook_audit"] = {"module_forward_pre_hooks": {
            "predict_calls": 2, "action_encode_calls": 2}}
    return {"cp10_goal_binding": {"root": "x"}, "model_before": {"x": "y"},
            "model_after": {"x": "y"}, "rows": [{"episode_id": episode,
                "native_success": True, "control_returned": 3, "encodings": [{}]}],
            "nodes": {episode: [{"boundary": 0, "status": "sealed", "receipt": receipt},
                                 {"boundary": 5, "status": "not_started", "receipt": None}]}}


def test_distinct_costs_remain_distinct():
    row = m.compare(fixture("structured"), fixture("cem"), 0)
    assert row["methods"][1]["charges"]["predict_calls"] == 2
    assert row["equal_total_compute"] is False
    assert "not charged" in row["methods"][1]["selection_accounting"]


@pytest.mark.parametrize("defect", ["goal", "model", "hooks", "boundary"])
def test_inconsistent_receipts_fail(defect):
    s, c = fixture("structured"), fixture("cem")
    if defect == "goal":
        c["cp10_goal_binding"] = {}
    elif defect == "model":
        c["model_after"] = {}
    elif defect == "hooks":
        s["nodes"]["s0_structured"][0]["receipt"]["module_hook_audit"]["module_forward_pre_hooks"]["predict_calls"] = 3
    else:
        c["nodes"]["s0_full_cem"][0]["boundary"] = 5
    with pytest.raises(ValueError):
        m.compare(deepcopy(s), deepcopy(c), 0)


def test_sealed_node_after_unstarted_node_fails():
    s, c = fixture("structured"), fixture("cem")
    nodes = c["nodes"]["s0_full_cem"]
    nodes.insert(0, {"boundary": 5, "status": "not_started", "receipt": None})
    with pytest.raises(ValueError, match="sealed node after unstarted"):
        m.compare(s, c, 0)
