import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


n = load("native_meter", ROOT / "experiments/spectral_reachability/codex_native_runtime_meter_20261030.py")
t = load("online_test_helpers", ROOT / "tests/test_codex_online_model_action_20261030.py")


class Encoder:
    def forward(self, data):
        return data
    def __call__(self, data):
        return self.forward(data)


class Model:
    def __init__(self):
        self.action_encoder = Encoder()
    def predict(self, state, actions):
        return state


def inventory(tmp_path):
    path = tmp_path / "native_source_fixture"
    path.write_bytes(b"fixture")
    return {"fixture": {"path": str(path), "sha256": n.digest(path)}}


def test_sra_native_seam_and_saved_journal(tmp_path):
    j, sources, method = t.setup(tmp_path, cap=3000)
    model = Model()
    source = inventory(tmp_path)
    original_predict = model.predict.__func__
    def native(model, device, ready, compiled, replay, emit):
        assert ready["root"]["context"] == t.encoder(t.FRAMES, [t.PACKET, t.PACKET])
        actions = model.action_encoder(np.zeros((1, 3, 10)))
        model.predict(np.zeros((1, 3, 192)), actions)
        emit({"kind": "jacobian", "completed": True})
        return {"fixture": True}, [{"completed": True}]
    def planner(request, context, meter):
        output = n.sra_online_bank(model, "fixture", context, {}, None, meter,
                                   native=native, source_inventory=source)
        assert [e["kind"] for e in output["native_entries"]] == ["action_encoder", "predict"]
        assert output["operation_events"][0]["completed"]
        assert output["resources_are_complete"] is False
        result = t.planner(request, context, meter)
        return result
    out = t.select(j, sources, method, tmp_path, plan=planner)
    assert model.predict.__func__ is original_predict and "predict" not in vars(model)
    j.call("stop", t.A, "fixture")
    j.call("stop", t.B, "fixture")
    sha = j.finish()
    consumed = t.m.consume_online(j.path, sha, [(tmp_path / (t.A + ".json"), out["receipt_sha256"])])
    row = consumed["journal"]["snapshot"]["rows"][0]
    assert row["resources_attempted"]["jacobian"] == 1
    assert row["resources_attempted"]["predict_calls"] == 2
    events = [json.loads(line) for line in j.path.read_bytes().splitlines()]
    assert any(e["kind"] == "begin" and e["payload"]["operation"] == "record_resource_attempt"
               and ":native-stage:sra" in e["payload"]["args"][1] for e in events)


def test_full_component_uses_live_context_and_meter(tmp_path, monkeypatch):
    j, sources, method = t.setup(tmp_path, cap=3000)
    model = Model()
    seen = []
    def fake_native(model, device, context, compiled):
        seen.append((context, compiled))
        model.predict(np.zeros((2, 3, 192)), model.action_encoder(np.zeros((2, 3, 10))))
        return {"fixture_only": True}
    monkeypatch.setattr(n, "_full_native", fake_native)
    def planner(request, context, meter):
        result = n.full_packet_online_compile(model, None, context, {"goal": "fixture"}, meter,
                                              source_inventory=inventory(tmp_path))
        assert result["native_entries"][1]["batch"] == 2
        assert not result["equal_total_compute"] and result["flops"] is None
        return t.planner(request, context, meter)
    t.select(j, sources, method, tmp_path, plan=planner)
    assert seen[0][0] == t.encoder(t.FRAMES, [t.PACKET, t.PACKET])
    assert j.protocol.state(t.A)["attempted"]["vjp_directions"] == 2112
    j.call("stop", t.A, "fixture")
    j.call("stop", t.B, "fixture")
    j.finish()


def test_native_failure_preserves_pending_and_restores_methods(tmp_path):
    j, sources, method = t.setup(tmp_path, cap=3000)
    model = Model()
    def native(model, device, ready, compiled, replay, emit):
        model.predict(np.zeros((1, 3, 192)), None)
        raise RuntimeError("native failure")
    def planner(request, context, meter):
        return n.sra_online_bank(model, None, context, {}, None, meter,
                                 native=native, source_inventory=inventory(tmp_path))
    with pytest.raises(RuntimeError, match="native failure"):
        t.select(j, sources, method, tmp_path, plan=planner)
    assert j.protocol.state(t.A)["phase"] == "stopped"
    assert len(j.protocol.state(t.A)["pending_resources"]) == 1
    assert "predict" not in vars(model) and "forward" not in vars(model.action_encoder)
    j.call("stop", t.B, "fixture")
    j.finish()


def test_budget_exhaustion_blocks_actual_entry(tmp_path):
    j, sources, method = t.setup(tmp_path, cap=3000)
    # Fault injection on this in-memory instance; the partial journal is not consumed.
    j.protocol.caps[t.A]["predict_calls"] = 0
    j.config["caps"][t.A]["predict_calls"] = 0
    called = []
    model = Model()
    model.predict = lambda *args: called.append(True)
    def planner(request, context, meter):
        def native(model, device, ready, compiled, replay, emit):
            model.predict(np.zeros((1, 3, 192)), None)
        return n.sra_online_bank(model, None, context, {}, None, meter,
                                 native=native, source_inventory=inventory(tmp_path))
    with pytest.raises(ValueError, match="budget exhausted"):
        t.select(j, sources, method, tmp_path, plan=planner)
    assert not called and j.protocol.state(t.A)["pending_resources"]
    j.close_partial()


def test_rgb_bytes_and_source_drift(tmp_path):
    seen = []
    def native(model, device, frames, actions, p0, transform):
        seen.append((frames, actions))
        return "context"
    encode = n.native_rgb_encoder(None, None, None, None, (2, 3, 3), native=native)
    assert encode([bytes(range(18))] * 3, [t.PACKET, t.PACKET]) == "context"
    assert seen[0][0][0].shape == (2, 3, 3) and seen[0][0][0].dtype == np.uint8
    with pytest.raises(ValueError, match="frame size"):
        encode([b"x"] * 3, [])
    src = inventory(tmp_path)
    Path(src["fixture"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="inventory"):
        n.verify_files(src)
