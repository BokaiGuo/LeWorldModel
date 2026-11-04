from contextlib import contextmanager
import json

import pytest

from experiments.spectral_reachability import p2_same_bank_env_repair_20261102_exclusive as repair


def test_real_bindings_cpu_only():
    plan = repair.build_plan()
    assert plan["technical_plan_sha256"] == repair.OLD_PLAN_SHA
    assert not plan["independent_confirmation"]
    assert not plan["equal_total_compute"]


def test_bad_authorization_never_dispatches(monkeypatch):
    monkeypatch.setattr(repair, "verify", lambda *_: pytest.fail("must reject before verify"))
    with pytest.raises(ValueError, match="authorization"):
        repair.run("unused", "wrong")


def test_failed_preflight_records_and_never_loads(tmp_path, monkeypatch):
    @contextmanager
    def lock(_):
        yield
    monkeypatch.setattr(repair.core.native.locking, "gpu_lock", lock)
    monkeypatch.setattr(repair.core.native, "native_session",
                        lambda *_: pytest.fail("model load must not run"))
    with pytest.raises(RuntimeError, match="preflight"):
        with repair.repaired_session(None, None, tmp_path, "fixture", lambda: (False, 0)):
            pytest.fail("must not yield")
    witness = json.loads((tmp_path / "environment.json").read_text())
    assert witness["status"] == "preflight_failed"
    assert witness["repair_plan_sha256"] == "fixture"


def test_probe_lock_released_before_native_lock(tmp_path, monkeypatch):
    events = []
    @contextmanager
    def lock(_):
        events.append("lock")
        yield
        events.append("unlock")
    @contextmanager
    def native(*_):
        events.append("native")
        yield ("callback", "meter")
    monkeypatch.setattr(repair.core.native.locking, "gpu_lock", lock)
    monkeypatch.setattr(repair.core.native, "native_session", native)
    with repair.repaired_session(None, None, tmp_path, "fixture", lambda: (True, 1)) as result:
        assert result == ("callback", "meter")
    assert events == ["lock", "unlock", "native"]
    assert json.loads((tmp_path / "environment.json").read_text())["cuda_device_count"] == 1


def test_output_collision_blocks_dispatch(monkeypatch):
    monkeypatch.setattr(repair, "verify", lambda *_: None)
    monkeypatch.setattr(repair.os.path, "lexists", lambda *_: True)
    with pytest.raises(ValueError, match="already exists"):
        repair.run("fixture", repair.AUTH)
