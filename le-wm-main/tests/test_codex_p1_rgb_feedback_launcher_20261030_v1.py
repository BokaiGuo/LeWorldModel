import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from experiments.spectral_reachability import codex_p1_rgb_feedback_episode_20261030_v1 as ep
from experiments.spectral_reachability import codex_p1_rgb_feedback_saved_20261030_v1 as saved
from experiments.spectral_reachability import codex_p1_rgb_feedback_launcher_20261030_v1 as launcher


@pytest.mark.parametrize("stop_step,reject,expected_plans", [
    (None, False, [1, 2]), (1, False, [1]), (5, False, [1]),
    (10, False, [1, 2]), (None, True, [1, 2])])
def test_phase_order_and_native_stop_without_physics(tmp_path, monkeypatch, stop_step, reject, expected_plans):
    image = np.full((224, 224, 3), (65, 105, 225), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    png = buf.getvalue()
    (tmp_path/"cp5.png").write_bytes(png)
    pixel_sha = hashlib.sha256(image.tobytes()).hexdigest()
    file_sha = hashlib.sha256(png).hexdigest()
    class Store:
        path = tmp_path
        events = []
        def emit(self, kind, **kw):
            self.events.append({"kind": kind, "index": len(self.events), **kw})
    store = Store()
    body = SimpleNamespace(eval_state=lambda goal, state: (False,))
    env = SimpleNamespace(unwrapped=body)
    meter = SimpleNamespace(where={})
    row = {"trajectory": [], "responses": []}
    anchor = {"index": 0, "condition": "off", "prefix5_float32_hex": np.zeros((5, 2), "<f4").tobytes().hex(),
              "checkpoint": {"pixel_sha256": pixel_sha}}
    packet = {"rgb_history": np.stack([image]*3), "raw_action_history": np.zeros((10, 2), "<f4"),
              "goal_rgb": image}
    root = {"packets": {"off": packet}, "slot": 0, "warm": [None]*11,
            "goal": np.zeros(7), "root_sha256": "root"}
    monkeypatch.setattr(ep.prior, "TraceStore", lambda s: s)
    monkeypatch.setattr(ep.prior, "compare_trace", lambda row, anchor, trace, meter:
                        trace.emit("suffix_start_verified"))
    monkeypatch.setattr(ep.base, "place", lambda *a: ({"state": np.zeros(7)}, np.zeros(7)))
    monkeypatch.setattr(ep.base, "capture", lambda env, obs, phase, step, row, meter, trace, *a:
                        (row["trajectory"].append({"png": "cp5.png", "file_sha256": file_sha,
                                                   "pixel_sha256": pixel_sha}) or image))
    def dispatch(env, cmd, phase, step, row, meter, trace, goal):
        return ({"state": np.zeros(7)}, {"goal": image}, phase == "probe" and step == stop_step,
                False, 0., cmd)
    monkeypatch.setattr(ep.base, "dispatch", dispatch)
    monkeypatch.setattr(ep.base.paired, "check_return", lambda *a: None)
    monkeypatch.setattr(ep.base.moving, "warmup_commands", lambda: np.zeros((10, 2), "<f4"))
    monkeypatch.setattr(ep.base.moving, "exact", np.array_equal)
    monkeypatch.setattr(ep.base.source, "vector", lambda x: np.asarray(x))
    monkeypatch.setattr(ep.base.source, "root_hash", lambda *a: "root")
    monkeypatch.setattr(ep.base, "bind_goal", lambda *a: None)
    monkeypatch.setattr(ep.base, "invariant", lambda *a: {"snapshot": {}})
    monkeypatch.setattr(ep.journal_feedback, "from_journal",
                        lambda *a, **kw: {"png": png, "png_sha256": file_sha, "pixel_sha256": pixel_sha,
                                          "last_render_index": 0})
    monkeypatch.setattr(ep.rgb_bridge, "decide", lambda events, folder, idx, cond, previous, after,
                        goal, digest, model: {"decision": "rgb_rejected" if reject else "model_packet",
                                              "executed_packet_f32_hex": None if reject else model.tobytes().hex()})
    calls = []
    def planner(number, supplied_png, executed):
        calls.append(number)
        assert supplied_png == png
        assert (executed is None) == (number == 1)
        horizon = np.full((2, 5, 2), number/10, dtype="<f4")
        if number == 2:
            assert np.all(executed == np.float32(.1))
        return {"raw_horizon_f32_hex": horizon.tobytes().hex(), "execute_packet": 0,
                "stage_report_sha256": "a"*64}
    ep.episode(env, root, None, anchor, row, meter, store, planner)
    assert calls == expected_plans
    assert row["status"] == ("rgb_rejected" if reject else
                             "probe_completed" if stop_step is None else "native_stopped")
    assert len(row["responses"]) == (5 if reject else 10 if stop_step is None else stop_step)


def test_bad_horizon_rejected():
    with pytest.raises(ValueError):
        ep._packet({"execute_packet": 0, "raw_horizon_f32_hex": "00"*40})


def test_consumer_rejects_partial_before_reading_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(saved.run, "verify", lambda *args: {})
    (tmp_path/"run_report.json").write_text('{"status":"partial_or_failed","prepare_sha256":"x"}')
    digest = saved.sha(tmp_path/"run_report.json")
    with pytest.raises((ValueError, KeyError)):
        saved.consume(tmp_path, digest, "ignored", "x")


def test_prepare_fixes_complete_roster_and_goal_sources():
    prepared = launcher.build()
    assert prepared["rows"] == len(prepared["roster"]) == 24
    assert [r["index"] for r in prepared["roster"]] == list(range(24))
    assert all(len(r["goal_rgb_pixel_sha256"]) == 64 for r in prepared["roster"])
    assert not prepared["equal_budget_certified"] and not prepared["P1_admission"]
    assert str(Path(launcher.episode.__file__).resolve()) in prepared["sources"]
    assert str(Path(saved.__file__).resolve()) in prepared["sources"]


def test_constructor_failure_is_durable_and_not_success(tmp_path):
    anchor = launcher.h10.design.verify(launcher.h10.INPUT, launcher.h10.INPUT_SHA)["anchors"][0]
    def fail():
        raise RuntimeError("fixture constructor failure")
    report = launcher.execute_row(anchor, {}, {}, tmp_path/"row", None, None, None, {}, fail, 0)
    assert report["status"] == "partial_or_failed"
    assert report["row"]["status"] == "engineering_failed"
    assert report["errors"] and "fixture constructor failure" in json.dumps(report["errors"])
    saved_report = json.loads((tmp_path/"row/physical/report.json").read_bytes())
    assert saved_report["status"] == "partial_or_failed"
    assert (tmp_path/"row/physical/journal.jsonl").is_file()
