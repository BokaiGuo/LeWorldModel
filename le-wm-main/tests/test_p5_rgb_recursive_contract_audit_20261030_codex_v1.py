import copy

import numpy as np
import pytest

from experiments.spectral_reachability import p5_rgb_recursive_contract_audit_20261030_codex_v1 as m


def output(x):
    a = np.asarray([x, 0, 0, 0, 0, 0], dtype="<f4")
    return {"values": a.tolist(), "sha256": m.bridge.digest(a.tobytes())}


def fixture():
    op, ov, rp, rv = [], [], [], []
    actions = np.zeros((5, 2), dtype="<f4")
    for i in range(164):
        ident = {"index": i, "root_slot": i // 82, "condition": ("off", "on")[(i // 41) % 2]}
        inputs = {**ident, "task_id": str(i), "selection_digest": str(i)}
        image = {"hash": "image"}
        op.append({**inputs, "actions": actions.tolist(), "actions_sha256": m.bridge.digest(actions.tobytes()),
                   "previous_rgb": image, "current_rgb": image})
        ov.append({**ident, "encode": {"output": output(0)},
                   "steps": [{"step": h, "status": "returned", "output": output(h)} for h in range(1, 6)]})
        steps = [{"horizon": h, "previous_plan_step": h + 3, "current_plan_step": h + 4,
                  "target_plan_step": h + 5, "previous_rgb": image, "current_rgb": image,
                  "action": actions[h-1].tolist(), "action_sha256": m.bridge.digest(actions[h-1].tobytes())}
                 for h in range(1, 6)]
        rp.append({**inputs, "steps": steps})
        rv.append({**ident, "steps": [{"horizon": h, "encode_state": output(0), "predict_next": output(1)}
                                      for h in range(1, 6)]})
    norm = {"mean": [0.] * 6, "std": [1.] * 6}
    return ({"rows": op, "normalization_from_prior_strict_load": norm},
            {"rows": ov, "journal_sha256": "old"},
            {"rows": rp, "normalization_from_prior_strict_load": copy.deepcopy(norm)},
            {"rows": rv, "journal_sha256": "new"})


def test_fixed_input_and_output_differences():
    out = m.analyze(*fixture())
    for condition in ("off", "on"):
        s = out["summary"][condition]["horizons"]
        assert s[0]["input_exact_equal_count"] == s[0]["output_exact_equal_count"] == 82
        assert s[4]["mean_input_difference_l2_normalized"] == 4
        assert s[4]["mean_output_difference_l2_normalized"] == 4
        assert s[4]["mean_abs_input_difference_normalized"] == [4, 0, 0, 0, 0, 0]


@pytest.mark.parametrize("mutation", ["norm", "identity", "action", "frame", "time", "output", "missing"])
def test_fail_closed(mutation):
    a, b, c, d = copy.deepcopy(fixture())
    if mutation == "norm":
        c["normalization_from_prior_strict_load"]["std"][0] = 2
    elif mutation == "identity":
        c["rows"][5]["task_id"] = "different"
    elif mutation == "action":
        c["rows"][5]["steps"][3]["action"][0] = 1
    elif mutation == "frame":
        c["rows"][5]["steps"][0]["current_rgb"] = {"hash": "different"}
    elif mutation == "time":
        c["rows"][5]["steps"][3]["target_plan_step"] = 6
    elif mutation == "output":
        d["rows"][5]["steps"][3]["predict_next"]["values"][0] = 2
    else:
        b["rows"].pop()
    with pytest.raises(ValueError):
        m.analyze(a, b, c, d)
