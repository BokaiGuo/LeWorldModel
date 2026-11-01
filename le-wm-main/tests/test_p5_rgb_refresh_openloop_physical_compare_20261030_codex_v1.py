import copy
import math

import numpy as np
import pytest

from experiments.spectral_reachability import p5_rgb_refresh_openloop_physical_compare_20261030_codex_v1 as m


def fixture():
    old_rows, old_predictions, refresh_rows, refresh_predictions, physical_rows, panel_rows, pairs = ([] for _ in range(7))
    actions = np.zeros((10, 2), dtype="<f4")
    for i in range(164):
        condition = ("off", "on")[(i // 41) % 2]
        ident = dict(index=i, root_slot=i // 82, condition=condition, task_id=str(i), selection_digest=str(i))
        frames = []
        for step in range(4, 11):
            state = [float(i), 0., 2., 3., 0., 0., 0.]
            frames.append(dict(phase="plan", step=step, condition=condition, overlay_index=int(condition == "on"),
                               png=f"b{i}_{step}.png", file_sha256=f"f{i}_{step}", pixel_sha256=f"p{i}_{step}",
                               state=state, snapshot={"state": list(state)}))
        def rec(step):
            frame = frames[step - 4]
            return {"path": "/x/" + frame["png"], "file_sha256": frame["file_sha256"],
                    "pixel_sha256": frame["pixel_sha256"]}
        plan = {"commands": actions.tolist()}
        physical_rows.append({**ident, "status": "horizon_completed", "own_returned": 10,
                              "initial_exact": True, "plan": plan, "trajectory": frames})
        panel_rows.append({**ident, "raw10_sha256": m.bridge.digest(actions.tobytes())})
        old_rows.append({**ident, "previous_rgb": rec(4), "current_rgb": rec(5),
                         "actions": actions[5:].tolist(), "actions_sha256": m.bridge.digest(actions[5:].tobytes())})
        vec = np.asarray([float(i) + 2, 0., 2., 3., 0., 1.], dtype="<f4")
        old_predictions.append({"index": i, "root_slot": i // 82, "condition": condition,
                                "steps": [{"step": h, "status": "returned", "output": {"values": vec.tolist()}}
                                          for h in range(1, 6)]})
        steps, preds = [], []
        for h in range(1, 6):
            act = actions[4 + h]
            steps.append({"horizon": h, "previous_plan_step": h + 3, "current_plan_step": h + 4,
                          "target_plan_step": h + 5, "previous_rgb": rec(h + 3), "current_rgb": rec(h + 4),
                          "action": act.tolist(), "action_sha256": m.bridge.digest(act.tobytes())})
            vec = np.asarray([float(i) + 1, 0., 2., 3., 0., 1.], dtype="<f4")
            preds.append({"horizon": h, "predict_next": {"values": vec.tolist(), "sha256": m.bridge.digest(vec.tobytes())}})
        refresh_rows.append({**ident, "steps": steps})
        refresh_predictions.append({"index": i, "root_slot": i // 82, "condition": condition, "steps": preds})
    for slot in range(4):
        for local in range(20):
            pairs.append(dict(slot=slot, root_slot=slot // 2, condition=("off", "on")[slot % 2],
                              minus_index=slot * 41 + 1 + local * 2, plus_index=slot * 41 + 2 + local * 2))
    norm = {"mean": [0.] * 6, "std": [1.] * 6}
    source = {"checkpoint": m.bridge.FILES["checkpoint"][1]}
    oldp = {"rows": old_rows, "pairs": pairs, "normalization_from_prior_strict_load": norm, "source_sha256": source}
    newp = {"rows": refresh_rows, "pairs": pairs, "normalization_from_prior_strict_load": copy.deepcopy(norm),
            "source_sha256": dict(source)}
    oldr = {"rows": old_predictions, "journal_sha256": "old"}
    newr = {"rows": refresh_predictions, "journal_sha256": "new"}
    phys = {"rows": physical_rows, "fixed_denominator": 164, "status": "observed_raw164_physical_recorded_not_admitted"}
    panel = {"rows": panel_rows, "pairs": pairs, "branch_denominator": 164, "pair_denominator": 80}
    return oldp, oldr, newp, newr, phys, panel


def test_six_axes_full_denominator_and_sign():
    result = m.compare(*fixture())
    assert len(result["rows"]) == 164
    for group in ("off", "on"):
        assert result["summary"][group]["denominator"] == 82
        for horizon in result["summary"][group]["horizons"]:
            assert horizon["open_absolute_error"] == [2., 0., 0., 0., 0., 0.]
            assert horizon["refresh_absolute_error"] == [1., 0., 0., 0., 0., 0.]
            assert horizon["refresh_minus_open_absolute_error"] == [-1., 0., 0., 0., 0., 0.]


@pytest.mark.parametrize("mutation", ["norm", "checkpoint", "identity", "frame", "action", "time", "output",
                                      "nonfinite", "physical", "missing", "pair"])
def test_reject_misaligned_inputs(mutation):
    a, b, c, d, e, f = copy.deepcopy(fixture())
    if mutation == "norm":
        c["normalization_from_prior_strict_load"]["mean"][0] = 1
    elif mutation == "checkpoint":
        c["source_sha256"]["checkpoint"] = "wrong"
    elif mutation == "identity":
        c["rows"][5]["task_id"] = "wrong"
    elif mutation == "frame":
        c["rows"][5]["steps"][1]["current_rgb"]["pixel_sha256"] = "wrong"
    elif mutation == "action":
        c["rows"][5]["steps"][1]["action"][0] = 1
    elif mutation == "time":
        c["rows"][5]["steps"][1]["target_plan_step"] = 9
    elif mutation == "output":
        d["rows"][5]["steps"][1]["predict_next"]["sha256"] = "wrong"
    elif mutation == "nonfinite":
        d["rows"][5]["steps"][1]["predict_next"]["values"][0] = math.nan
    elif mutation == "physical":
        e["rows"][5]["trajectory"][-1]["state"][0] = 12
    elif mutation == "missing":
        c["rows"].pop()
    else:
        f["pairs"][0]["minus_index"] = 3
    with pytest.raises(ValueError):
        m.compare(a, b, c, d, e, f)


def test_bad_seal_stops_before_physical_read(monkeypatch):
    def fail():
        raise ValueError("invalid seal")
    def forbidden(*args):
        pytest.fail("physical receipt read after seal failure")
    monkeypatch.setattr(m, "sealed_predictions", fail)
    monkeypatch.setattr(m.bridge, "read_fixed", forbidden)
    with pytest.raises(ValueError, match="invalid seal"):
        m.build()
