import copy
from contextlib import contextmanager

import pytest

from experiments.spectral_reachability import p3_all30_localguard_cpu_20261102_codexexclusive as local


def test_fixed_old_roster_caps_and_new_denominator():
    p, _ = local.material()
    prior, _ = local.old.verify(local.OLD_PREP, local.OLD_SHA)
    assert p["plans"] == prior["plans"]
    assert p["contract"]["caps"] == prior["contract"]["caps"]
    assert p["contract"]["no_retries"]
    assert "no imputation" in p["contract"]["denominator_rule"]


def test_only_expected_guard_is_local():
    row = {"status": "failed", "initial_exact": True, "secondary_errors": [],
           "first_error": {"stage": "branch", "type": "ValueError", "message": local.GUARD_ERROR},
           "own_returned": 8}
    guard = {"branch": 7, "phase": "plan", "accepted": False,
             "render_preserved_dynamics": True, "step": 8}
    assert local.local_guard(row, guard, 7)
    for change in ({"phase": "warmup"}, {"accepted": True}, {"branch": 8},
                   {"render_preserved_dynamics": False}, {"step": 9}):
        bad = dict(guard, **change)
        assert not local.local_guard(row, bad, 7)
    for change in ({"initial_exact": False}, {"secondary_errors": [{"message": "close"}]},
                   {"first_error": {"stage": "branch", "type": "ValueError", "message": "disk"}}):
        bad = copy.deepcopy(row)
        bad.update(change)
        assert not local.local_guard(bad, guard, 7)


def test_gpu_not_authorized(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="CPU authorization"):
        local.run("unused", "unused", "unused", local.AUTH)


def test_local_failure_does_not_skip_or_retry_other_candidates(monkeypatch, tmp_path):
    p, data = local.material()
    p["bindings"] = {}
    store = local.Store(tmp_path / "run")
    visited = []
    class Body:
        render_action = False
        render_size = 224
        render_mode = "rgb_array"
    class Env:
        unwrapped = Body()
        def close(self):
            pass
    @contextmanager
    def hooks(env, ops, row):
        try:
            yield
        except Exception as exc:
            local.moving.retain(row, exc, "branch")
    def branch(env, ref, source, plan, row, ops, output, index):
        visited.append(index)
        row.update(initial_exact=True, own_returned=8)
        if index == 0:
            store.latest_guard = {"branch": 0, "phase": "plan", "accepted": False,
                                  "render_preserved_dynamics": True, "step": 8}
            raise ValueError(local.GUARD_ERROR)
        row["status"] = "horizon_completed"
    monkeypatch.setattr(local.moving.moving, "validate_env", lambda _: None)
    monkeypatch.setattr(local.moving, "native_hooks", hooks)
    monkeypatch.setattr(local.moving, "branch", branch)
    report = local.execute(p, data, store, Env)
    assert visited == list(range(30))
    assert report["rows"][0]["status"] == "guard_failed_invalid"
    assert report["candidate_counts"]["guard_failed_invalid"] == 1
    assert report["candidate_counts"]["horizon_completed"] == 29
    assert report["status"] == "partial_or_failed"  # simulated callbacks lack real resource receipts
