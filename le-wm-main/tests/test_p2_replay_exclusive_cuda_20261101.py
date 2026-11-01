import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import p2_replay_exclusive_cuda_20261101 as m


def test_lock_is_project_cooperative_lock():
    assert m.LOCK == m.frontier.PROJECT / "results/e3_p2_host_cuda_cooperative.lock"


def test_census_rejects_busy_and_unknown_device(monkeypatch):
    def run(args, **kwargs):
        text = ("GPU-abc, Test, 0, 100\n" if "--query-gpu=uuid,name,memory.used,memory.total" in args
                else "GPU-abc, 123, 50\n")
        return SimpleNamespace(stdout=text)
    monkeypatch.setattr(m.subprocess, "run", run)
    with pytest.raises(ValueError, match="active"):
        m.census("GPU-abc")
    with pytest.raises(ValueError, match="UUID"):
        m.census("GPU-missing")


def test_failure_is_durable_and_output_cannot_be_reused(tmp_path):
    out = tmp_path / "attempt"
    with pytest.raises(ValueError, match="coordination"):
        m.execute(inventory="x", inventory_sha256="x", output=out,
                  device_uuid="GPU-a", coordinated=False)
    assert json.loads((out / "failure.json").read_text())["physical_authorized"] is False
    with pytest.raises(FileExistsError):
        m.execute(inventory="x", inventory_sha256="x", output=out,
                  device_uuid="GPU-a", coordinated=False)


def test_locked_source_recheck_failure_never_reaches_census(tmp_path, monkeypatch):
    calls = []
    def verify(*_):
        calls.append("verify")
        if calls.count("verify") == 2:
            raise ValueError("drift")
        return {"prepared": "p", "prepared_sha256": "h"}
    @contextmanager
    def lock(path):
        calls.append("lock")
        try:
            yield
        finally:
            calls.append("unlock")
    monkeypatch.setattr(m.seal, "verify", verify)
    monkeypatch.setattr(m.frontier, "load_prepared", lambda *_: (None, {}, {}))
    monkeypatch.setattr(m.locking, "gpu_lock", lock)
    monkeypatch.setattr(m, "census", lambda _: pytest.fail("census reached"))
    out = tmp_path / "run"
    # Count source verifications separately from lock event.
    calls.clear()
    with pytest.raises(Exception):
        m.execute(inventory="x", inventory_sha256="h", output=out,
                  device_uuid="GPU-a", coordinated=True)
    assert (out / "failure.json").exists()
    assert "unlock" in calls


def test_census_failure_records_stage_and_runs_postcheck(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(m.seal, "verify", lambda *_: calls.append("verify") or
                        {"prepared": "p", "prepared_sha256": "h"})
    monkeypatch.setattr(m.frontier, "load_prepared", lambda *_: (None, {}, {}))
    @contextmanager
    def lock(_):
        calls.append("lock")
        try:
            yield
        finally:
            calls.append("unlock")
    monkeypatch.setattr(m.locking, "gpu_lock", lock)
    def fail(_):
        raise ValueError("busy")
    monkeypatch.setattr(m, "census", fail)
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="busy"):
        m.execute(inventory="x", inventory_sha256="h", output=out,
                  device_uuid="GPU-a", coordinated=True)
    failure = json.loads((out / "failure.json").read_text())
    assert failure["stage"] == "device_census"
    assert failure["checks"]["sources_unchanged"] is True
    assert calls == ["verify", "lock", "verify", "verify", "unlock"]
