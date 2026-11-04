from contextlib import nullcontext

import pytest

from experiments.spectral_reachability import p3_all30_cpu_physical_20261102_codexexclusive as replay


def test_prepare_binds_all_30_and_caps():
    p, data = replay.material()
    assert [(x["root_slot"], x["candidate_index"]) for x in p["plans"]] == [
        (r, i) for r in (0, 1) for i in range(15)]
    assert len(data["references"]) == 2
    assert p["contract"]["caps"]["public_step"] == 600
    assert p["contract"]["caps"]["physics_step"] == 6120
    assert p["contract"]["no_retries"]


def test_cuda_must_be_hidden(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="CPU authorization"):
        replay.run("unused", "unused", "unused", replay.AUTH)


def test_changed_candidate_is_rejected():
    with pytest.raises(ValueError, match="invalid action bits"):
        replay.audit.action([[float("nan"), 0]] * 10)


def test_guard_failure_stops_remaining_branches(monkeypatch, tmp_path):
    p, data = replay.material()
    p["bindings"] = {}
    class Body:
        render_action = False
        render_size = 224
        render_mode = "rgb_array"
    class Env:
        unwrapped = Body()
        def close(self):
            pass
    def failure(*args):
        raise ValueError("simulated root guard failure")
    monkeypatch.setattr(replay.moving.moving, "validate_env", lambda _: None)
    monkeypatch.setattr(replay.moving, "native_hooks", lambda *args: nullcontext())
    monkeypatch.setattr(replay.moving, "branch", failure)
    store = replay.Store(tmp_path / "test")
    report = replay.execute(p, data, store, Env)
    assert report["status"] == "partial_or_failed"
    assert report["rows"][0]["status"] == "failed"
    assert all(row["status"] == "not_started" for row in report["rows"][1:])
    assert (store.path / "journal.jsonl").exists()
