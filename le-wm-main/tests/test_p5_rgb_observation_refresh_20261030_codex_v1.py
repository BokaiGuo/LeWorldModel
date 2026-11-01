import copy

import numpy as np
import pytest

from experiments.spectral_reachability import p5_rgb_observation_refresh_20261030_codex_v1 as m


def fixture():
    old_rows, physical_rows, panel_rows, pairs = [], [], [], []
    actions = np.zeros((10, 2), dtype="<f4")
    for i in range(164):
        root, condition = i // 82, ("off", "on")[(i // 41) % 2]
        identity = dict(index=i, root_slot=root, condition=condition, task_id=str(i), selection_digest=str(i))
        records = [dict(phase="plan", step=s, condition=condition, overlay_index=int(condition == "on"),
                        png=f"b{i}_{s}.png", file_sha256=f"file{i}_{s}", pixel_sha256=f"pixel{i}_{s}")
                   for s in range(4, 10)]
        def image(f):
            return {"path": f["png"], "file_sha256": f["file_sha256"], "pixel_sha256": f["pixel_sha256"],
                    "model_input_chw_float32_sha256": f"input{f['png']}"}
        plan = {**identity, "commands": actions.tolist(), "raw_float32_sha256": m.digest(actions.tobytes())}
        old_rows.append({**identity, "actions_sha256": m.digest(actions[5:].tobytes()),
                         "previous_rgb": image(records[0]), "current_rgb": image(records[1])})
        physical_rows.append({**identity, "plan": plan, "trajectory": records, "status": "horizon_completed",
                              "own_returned": 10, "initial_exact": True})
        panel_rows.append({**identity, "raw10_sha256": plan["raw_float32_sha256"]})
    for slot in range(4):
        for local in range(20):
            pairs.append(dict(slot=slot, root_slot=slot // 2, condition=("off", "on")[slot % 2],
                              coord=local % 10, h=(0.01, 0.001)[local // 10],
                              minus_index=slot * 41 + 1 + 2 * local, plus_index=slot * 41 + 2 + 2 * local))
    old = dict(rows=old_rows, pairs=pairs, branch_denominator=164, pair_denominator=80,
               normalization_from_prior_strict_load={"mean": [0.] * 6, "std": [1.] * 6})
    physical = dict(rows=physical_rows, fixed_denominator=164, status="observed_raw164_physical_recorded_not_admitted")
    panel = dict(rows=panel_rows, pairs=pairs, branch_denominator=164, pair_denominator=80,
                 status="saved_input_physical_identity_bound_not_scientific_admission")
    return old, physical, panel, image


def test_refresh_time_and_fixed_denominators():
    old, phys, panel, image = fixture()
    result = m.prepare(old, phys, panel, image)
    assert len(result["rows"]) == 164 and len(result["pairs"]) == 80
    assert [(s["previous_plan_step"], s["current_plan_step"], s["target_plan_step"])
            for s in result["rows"][0]["steps"]] == [(4, 5, 6), (5, 6, 7), (6, 7, 8), (7, 8, 9), (8, 9, 10)]
    assert result["planned_calls"] == {"encode_state": 820, "predict_next": 820}


@pytest.mark.parametrize("mutation", ["identity", "action", "missing", "overlay", "pair", "history"])
def test_reject_bad_inputs(mutation):
    old, phys, panel, image = fixture()
    old, phys, panel = copy.deepcopy((old, phys, panel))
    if mutation == "identity":
        phys["rows"][0]["task_id"] = "bad"
    elif mutation == "action":
        phys["rows"][0]["plan"]["commands"][5][0] = 1
    elif mutation == "missing":
        phys["rows"][0]["trajectory"].pop()
    elif mutation == "overlay":
        phys["rows"][0]["trajectory"][2]["overlay_index"] = 1
    elif mutation == "pair":
        panel["pairs"][0]["plus_index"] = 17
    else:
        old["rows"][0]["current_rgb"]["file_sha256"] = "bad"
    with pytest.raises(ValueError):
        m.prepare(old, phys, panel, image)


class FakeBackend:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def metadata(self):
        return {"device": "synthetic"}

    def encode(self, inputs):
        self.calls.append(("encode", inputs))
        return np.zeros((1, 6), dtype="<f4")

    def step(self, inputs, index):
        self.calls.append(("step", inputs, index))
        if self.fail:
            raise RuntimeError("synthetic failure")
        return np.ones((1, 6), dtype="<f4")


class Journal:
    def __init__(self):
        self.events = []

    def emit(self, kind, **fields):
        self.events.append({"kind": kind, **fields})


def test_order_reencodes_each_observation_no_prediction_feedback():
    old, phys, panel, image = fixture()
    prepared = m.prepare(old, phys, panel, image)
    backend, journal = FakeBackend(), Journal()
    assert m.execute(prepared, "test", backend, journal, 100, clock=lambda: 0,
                     load=lambda step: step["current_plan_step"]) == "complete"
    assert len(backend.calls) == 1640
    assert backend.calls[:4] == [("encode", 5), ("step", 5, 0), ("encode", 6), ("step", 6, 0)]
    assert journal.events[-1]["status"] == "complete"
    assert len([e for e in journal.events if e["kind"] == "returned"]) == 1640


def test_failure_preserves_partial_and_no_retry():
    old, phys, panel, image = fixture()
    backend, journal = FakeBackend(True), Journal()
    assert m.execute(m.prepare(old, phys, panel, image), "test", backend, journal, 100,
                     clock=lambda: 0, load=lambda step: step["horizon"]) == "partial"
    assert len(backend.calls) == 2
    assert [e["kind"] for e in journal.events] == ["start", "attempt", "returned", "attempt", "failed", "end"]
    assert journal.events[-1]["status"] == "partial"


def test_exclusive_name():
    with pytest.raises(ValueError):
        m.target("../other.json", ".json")
