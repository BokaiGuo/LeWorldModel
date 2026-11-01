from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p1_backtrack_physical_episode_20261030 as mod


def setup_episode(monkeypatch, tmp_path, stop_at=None):
    warm = np.zeros((10, 2), np.float32)
    rgb = np.zeros((1,), np.uint8)
    body = SimpleNamespace(eval_state=lambda *_: (False,))
    env = SimpleNamespace(unwrapped=body)
    packet = {"raw_action_history": warm, "rgb_history": np.asarray([rgb] * 3), "goal_rgb": rgb}
    root = {"packets": {"off": packet}, "warm_trace": [], "slot": {}, "goal": np.zeros(7),
            "root_sha256": "b" * 64}
    plan = {"condition": "off", "task_id": "a" * 64, "physical_root_sha256": "b" * 64,
            "raw15_float32_hex": np.zeros((15, 2), "<f4").tobytes().hex()}
    row = {"index": 0, "condition": "off", "trajectory": [], "checkpoints": [],
           "missing_checkpoints": [0, 5, 10, 15], "prefix51": {"frames": []}, "responses": []}
    meter = SimpleNamespace(where={})
    store = SimpleNamespace(path=tmp_path, check=lambda *_: None, emit=lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(mod.native.moving, "warmup_commands", lambda: warm)
    monkeypatch.setattr(mod.native.moving, "image_sha", lambda _: "c" * 64)
    monkeypatch.setattr(mod.native.source, "root_hash", lambda *_: "b" * 64)
    monkeypatch.setattr(mod.native, "place", lambda *_: ({"state": np.zeros(7)}, np.zeros(7)))
    monkeypatch.setattr(mod.native, "bind_goal", lambda *_: None)
    monkeypatch.setattr(mod.native, "invariant", lambda *_: {"goal_pose": [0], "snapshot": {"state": [0] * 7}})
    monkeypatch.setattr(mod.native.paired, "check_return", lambda *_: False)
    monkeypatch.setattr(mod.native.base.previous, "metric", lambda *_: {"native_success": False, "native_distance": 1.0})

    def dispatch(_env, action, phase, step, *_):
        calls.append((phase, step))
        return ({"state": np.zeros(7)}, {"goal": rgb}, phase == "plan" and step == stop_at,
                False, -1.0, action.copy())

    def observe(_env, _obs, phase, step, row, *_):
        for i in range(3):
            receipt = {"phase": phase, "step": step, "condition": "on" if i == 1 else "off",
                       "overlay_index": i, "png": f"{step}-{i}.png", "file_sha256": "e" * 64,
                       "pixel_sha256": ("d" if i == 1 else "c") * 64,
                       "state": [0] * 7, "snapshot": {"state": [0] * 7}}
            if phase == "warmup" or step <= 5:
                row["prefix51"]["frames"].append({"receipt": receipt})
            if i == 0:
                chosen = receipt
        row["trajectory"].append(chosen)
        if phase == "plan" and step in (0, 5, 10):
            row["checkpoints"].append(chosen)
            row["missing_checkpoints"].remove(step)
        return rgb

    monkeypatch.setattr(mod.native, "dispatch", dispatch)
    monkeypatch.setattr(mod.native, "observe", observe)
    return env, root, plan, row, meter, store, calls


def test_adapter_executes_fixed_horizon_and_saves_cp15(monkeypatch, tmp_path):
    env, root, plan, row, meter, store, calls = setup_episode(monkeypatch, tmp_path)
    result = mod.episode(env, root, {}, plan, row, meter, store)
    assert result["status"] == "horizon_completed"
    assert [x["step"] for x in row["checkpoints"]] == [0, 5, 10, 15]
    assert len([x for x in calls if x[0] == "plan"]) == 15
    assert len(list(tmp_path.glob("cp5_*.json"))) == 1


def test_adapter_early_stop_does_not_create_future_checkpoints(monkeypatch, tmp_path):
    env, root, plan, row, meter, store, calls = setup_episode(monkeypatch, tmp_path, stop_at=6)
    result = mod.episode(env, root, {}, plan, row, meter, store)
    assert result["status"] == "native_stopped"
    assert row["missing_checkpoints"] == [10, 15]
    assert len([x for x in calls if x[0] == "plan"]) == 6


def test_adapter_rejects_changed_cp5_reference_before_suffix(monkeypatch, tmp_path):
    env, root, plan, row, meter, store, _ = setup_episode(monkeypatch, tmp_path)
    mod.episode(env, root, {}, plan, row, meter, store)
    reference = next(tmp_path.glob("cp5_*.json"))
    reference.write_bytes(b"tampered")
    env, root, plan, row, meter, store, calls = setup_episode(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="prefix reference mismatch"):
        mod.episode(env, root, {}, plan, row, meter, store)
    assert [step for phase, step in calls if phase == "plan"] == [1, 2, 3, 4, 5]
