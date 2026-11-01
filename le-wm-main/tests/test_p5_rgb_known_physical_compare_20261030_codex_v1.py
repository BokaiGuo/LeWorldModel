import copy

import numpy as np
import pytest

from experiments.spectral_reachability import p5_rgb_known_physical_compare_20261030_codex_v1 as c


def fixture():
    rows, physical, reviewed, bindings, pairs = [], [], [], [], []
    actions = np.zeros((10, 2), dtype="<f4")
    for i in range(164):
        root, condition = i // 82, ("off", "on")[(i // 41) % 2]
        identity = dict(index=i, root_slot=root, condition=condition, task_id=str(i), selection_digest=str(i))
        rows.append({**identity, "actions_sha256": c.bridge.digest(actions[5:].tobytes())})
        bindings.append({**identity, "raw10_sha256": c.bridge.digest(actions.tobytes())})
        state = [float(i), 0., 2., 3., 0., 0., 0.]
        trajectory = [dict(phase="plan", step=h + 5, condition=condition, overlay_index=int(condition == "on"),
                           state=state, snapshot={"state": state}) for h in range(1, 6)]
        physical.append({**identity, "status": "horizon_completed", "own_returned": 10,
                         "initial_exact": True, "plan": {"commands": actions.tolist()}, "trajectory": trajectory})
        reviewed.append({k: identity[k] for k in ("index", "root_slot", "condition")} |
                        {"steps": [{"step": h, "status": "returned",
                                    "output": {"values": [float(i) + 1, 0., 2., 3., 0., 1.]}}
                                   for h in range(1, 6)]})
    for slot in range(4):
        for local in range(20):
            pairs.append({"slot": slot, "minus_index": slot * 41 + 1 + local * 2,
                          "plus_index": slot * 41 + 2 + local * 2,
                          "condition": ("off", "on")[slot % 2], "root_slot": slot // 2})
    prepared = {"rows": rows, "pairs": pairs,
                "normalization_from_prior_strict_load": {"mean": [0.] * 6, "std": [1.] * 6}}
    review = {"rows": reviewed, "journal_sha256": "synthetic"}
    phys = {"rows": physical, "status": "observed_raw164_physical_recorded_not_admitted",
            "fixed_denominator": 164}
    panel = {"rows": bindings, "pairs": pairs, "branch_denominator": 164, "pair_denominator": 80}
    return prepared, review, phys, panel


def test_fixed_denominators_and_pair_cancellation():
    out = c.compare(*fixture())
    assert len(out["rows"]) == 164 and len(out["pairs"]) == 80
    for condition in ("off", "on"):
        for h in out["summary"][condition]["horizons"]:
            assert h["branch_coordinate_mae"] == [1., 0., 0., 0., 0., 0.]
            assert h["pair_coordinate_mae"] == [0.] * 6


@pytest.mark.parametrize("mutation", ["missing", "identity", "action", "target", "nonfinite", "pair", "norm"])
def test_fail_closed(mutation):
    p, r, ph, panel = copy.deepcopy(fixture())
    if mutation == "missing":
        ph["rows"].pop()
    elif mutation == "identity":
        ph["rows"][2]["task_id"] = "wrong"
    elif mutation == "action":
        ph["rows"][2]["plan"]["commands"][6][0] = 1
    elif mutation == "target":
        ph["rows"][2]["trajectory"].pop()
    elif mutation == "nonfinite":
        r["rows"][2]["steps"][0]["output"]["values"][0] = float("nan")
    elif mutation == "pair":
        panel["pairs"][0]["minus_index"] = 3
    else:
        p["normalization_from_prior_strict_load"]["std"][0] = 0
    with pytest.raises(ValueError):
        c.compare(p, r, ph, panel)


def test_no_label_read_when_seal_fails(monkeypatch):
    def fail():
        raise ValueError("bad seal")
    def forbidden(*args):
        pytest.fail("physical data read before seal")
    monkeypatch.setattr(c, "seal", fail)
    monkeypatch.setattr(c.bridge, "read_fixed", forbidden)
    with pytest.raises(ValueError, match="bad seal"):
        c.build()


def test_output_namespace():
    with pytest.raises(ValueError):
        c.target("../other.json")
