import hashlib
import subprocess
import sys
import threading

import pytest

from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import p2_durable_hook_driver_20261030_codex as driver
from experiments.spectral_reachability import p2_preregistered_resource_vector_20261030_codex as spec


def setup(tmp_path, cap=10, limit=driver.MAX_JOURNAL):
    h = "a" * 64
    c = {"schema": spec.NAME + ".contract.v1", "arms": ["joint", "composition"],
         "scope": "per_root_per_arm_standalone_planning_and_selected_physical_execution",
         "identity": {k: h for k in spec.IDENTITY},
         "axes": {k: {"unit": unit, "aggregation": "max" if k == "peak_device_bytes" else "sum",
                      "cap": cap} for k, unit in spec.AXES.items()}}
    digest = spec.sha(spec.canonical(c))
    obj = driver.Driver(tmp_path / "exclusive", c, digest, dict.fromkeys(core.AXES, 100),
                        "joint", "run-fixture", journal_limit=limit)
    return obj, digest


def verify(result, contract_sha):
    return driver.verify(result["journal"], result["sha256"], contract_sha)


def test_actual_durable_event_order_and_nested_recount(tmp_path):
    obj, digest = setup(tmp_path)
    def outer():
        raw = (obj.directory / "events.jsonl").read_bytes()
        assert driver.read_chain(raw)[-1]["payload"]["kind"] == "resource_attempt"
        return obj.call({"stage": "nested"}, "solve", lambda: 42)
    assert obj.call({"stage": "outer"}, "replay", outer) == 42
    result = obj.finish()
    checked = verify(result, digest)
    assert checked["status"] == "complete" and checked["events"] == 2
    assert checked["attempted"]["solve_calls"] == checked["completed"]["solve_calls"] == 1
    assert not checked["contract_receipt_ready"] and not checked["coverage_verified"]
    with pytest.raises(ValueError):
        obj.call({"stage": "late"}, "solve", lambda: None)


@pytest.mark.parametrize("resource,kwargs", [
    ("predict", {"shape": [1, 3, 192]}),
    ("vjp", {"directions": 1}), ("array_svd", {}), ("solve", {}), ("replay", {}),
])
def test_cap_refuses_before_callback_durably(tmp_path, resource, kwargs):
    obj, digest = setup(tmp_path, cap=0)
    called = []
    with pytest.raises(core.BudgetExceeded):
        obj.call({"stage": "fixture"}, resource, lambda: called.append(True), **kwargs)
    result = obj.finish()
    checked = verify(result, digest)
    assert called == [] and checked["status"] == "failed_or_blocked"
    events = driver.read_chain((obj.directory / "events.jsonl").read_bytes())
    assert events[1]["payload"]["kind"] == "resource_blocked"


def test_failed_work_remains_charged(tmp_path):
    obj, digest = setup(tmp_path)
    def fail():
        raise RuntimeError("fixture")
    with pytest.raises(RuntimeError):
        obj.call({"stage": "fixture"}, "solve", fail)
    with pytest.raises(ValueError, match="fail-stopped"):
        obj.call({"stage": "fixture"}, "solve", lambda: None)
    checked = verify(obj.finish(), digest)
    assert checked["status"] == "failed_or_blocked"
    assert checked["attempted"]["solve_calls"] == 1 and checked["completed"]["solve_calls"] == 0


def test_journal_limit_before_attempt_stops_callback_and_cannot_seal(tmp_path):
    obj, _ = setup(tmp_path)
    obj.journal.limit = obj.journal.size
    called = []
    with pytest.raises(ValueError, match="journal limit"):
        obj.call({"stage": "fixture"}, "solve", lambda: called.append(True))
    assert not called
    with pytest.raises(ValueError, match="journal fault"):
        obj.finish()
    raw = (obj.directory / "events.jsonl").read_bytes()
    with pytest.raises(ValueError, match="lifecycle"):
        driver.verify(obj.directory / "events.jsonl", hashlib.sha256(raw).hexdigest(), "a" * 64)


def test_fsync_failure_prevents_dispatch(tmp_path, monkeypatch):
    obj, _ = setup(tmp_path)
    called = []
    def bad_fsync(fd):
        raise OSError("fsync failure")
    monkeypatch.setattr(driver.transport.os, "fsync", bad_fsync)
    with pytest.raises(OSError, match="fsync failure"):
        obj.call({"stage": "fixture"}, "solve", lambda: called.append(True))
    assert not called
    with pytest.raises(ValueError, match="journal fault"):
        obj.finish()


def test_cli_rechecks_externally_anchored_journal(tmp_path):
    obj, digest = setup(tmp_path)
    obj.call({"stage": "fixture"}, "solve", lambda: None)
    result = obj.finish()
    completed = subprocess.run(
        [sys.executable, "-m", "experiments.spectral_reachability.p2_durable_hook_driver_20261030_codex",
         "--journal", result["journal"], "--sha256", result["sha256"], "--contract-sha256", digest],
        check=True, capture_output=True, text=True)
    assert '"contract_receipt_ready": false' in completed.stdout


def test_truncation_mutation_and_external_contract_rejected(tmp_path):
    obj, digest = setup(tmp_path)
    obj.call({"stage": "fixture"}, "solve", lambda: None)
    result = obj.finish()
    path = obj.directory / "events.jsonl"
    original = path.read_bytes()
    with pytest.raises(ValueError, match="commitment"):
        driver.verify(path, result["sha256"], "b" * 64)
    path.write_bytes(original[:-2])
    with pytest.raises(ValueError, match="SHA"):
        driver.verify(path, result["sha256"], digest)
    with pytest.raises(ValueError, match="incomplete"):
        driver.verify(path, hashlib.sha256(original[:-2]).hexdigest(), digest)
    changed = original.replace(b'"resource_attempt"', b'"resource_attempx"', 1)
    path.write_bytes(changed)
    with pytest.raises(ValueError, match="journal chain"):
        driver.verify(path, hashlib.sha256(changed).hexdigest(), digest)


def test_no_overwrite_and_no_finish_inside_callback(tmp_path):
    obj, digest = setup(tmp_path)
    with pytest.raises(ValueError, match="exclusive"):
        setup(tmp_path)
    def fn():
        with pytest.raises(ValueError, match="outside active"):
            obj.finish()
    obj.call({"stage": "fixture"}, "solve", fn)
    assert verify(obj.finish(), digest)["status"] == "complete"


def test_concurrent_calls_rejected(tmp_path):
    obj, digest = setup(tmp_path)
    entered, release = threading.Event(), threading.Event()
    errors = []
    def worker():
        try:
            obj.call({"stage": "thread"}, "solve", lambda: (entered.set(), release.wait(5)))
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=worker)
    thread.start()
    assert entered.wait(5)
    try:
        with pytest.raises(ValueError, match="concurrent"):
            obj.call({"stage": "other"}, "solve", lambda: None)
    finally:
        release.set()
        thread.join(5)
    assert not errors and not thread.is_alive()
    assert verify(obj.finish(), digest)["events"] == 1
