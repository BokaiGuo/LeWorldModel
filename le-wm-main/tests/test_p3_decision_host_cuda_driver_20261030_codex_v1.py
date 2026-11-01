"""CPU-only driver wiring tests; no model, CUDA, or physics execution."""
import hashlib
import json

import pytest

from experiments.spectral_reachability import p3_decision_host_cuda_driver_20261030_codex_v1 as d


def plan():
    return dict(gpu_uuid="GPU-12345678-1234-1234-1234-123456789abc",
                source_sha256="a" * 64, files={"adapter": "b" * 64},
                native_limits={}, native_cap=10000, native_reserve=100,
                decision_cap=10000, decision_reserve=100, outer_cap=1048576,
                floor=1, wall=60, allocated_cap=1, reserved_cap=2,
                coordination="main review")


def digest(p):
    return hashlib.sha256(d.control.core.canonical(p)).hexdigest()


class Decision:
    def __init__(self, path, **kwargs):
        path.mkdir()
        self.finished = self.poisoned = False

    def finish(self, roster):
        self.finished = True
        return "decision-hash"


class Native:
    def __init__(self, path, **kwargs):
        path.mkdir()
        (path / "events.jsonl").write_bytes(b"events\n")
        self.poisoned, self.pending, self.index, self.previous, self.used = False, {}, 1, "tail", {}


@pytest.fixture
def setup(monkeypatch):
    p = plan()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", p["gpu_uuid"])
    monkeypatch.setattr(d, "validate", lambda p: ({}, {"tasks": [1, 2]}, {}))
    monkeypatch.setattr(d.journal, "Store", Decision)
    monkeypatch.setattr(d.transport, "Store", Native)
    monkeypatch.setattr(d.transport, "inspect", lambda *a: {
        "transport_complete": True, "events": [{}], "tail": "tail", "pending": {}})
    return p


def roster(status="callback_complete_unverified"):
    return {"rows": [dict(slot=i, root=i // 3, policy="x", status=status,
                          cleanup="returned", error=None) for i in range(6)]}


def test_success_wires_prepared_and_journals(tmp_path, setup):
    def runtime(p, prepared, ds, ns):
        assert prepared["tasks"] == [1, 2]
        assert isinstance(ds, Decision) and isinstance(ns, Native)
        return {"before": "same", "after": "same"}, roster(), None
    result = d.run(tmp_path / "run", setup, digest(setup), d.AUTH, runtime=runtime)
    assert result["candidate_complete"] and not result["scientific_pass"]
    assert len(result["rows"]) == 6
    assert result["native_files"][0]["sha256"] == hashlib.sha256(b"events\n").hexdigest()
    assert json.loads((tmp_path / "run" / "begin.json").read_text())["plan"] == setup


def test_failed_roster_is_terminal_and_failure(tmp_path, setup):
    def runtime(*args):
        return None, roster("setup_failed"), "adapter poisoned"
    with pytest.raises(RuntimeError, match="incomplete"):
        d.run(tmp_path / "run", setup, digest(setup), d.AUTH, runtime=runtime)
    end = json.loads((tmp_path / "run" / "end.json").read_text())
    failure = json.loads((tmp_path / "run" / "failure.json").read_text())
    assert not end["candidate_complete"] and len(end["rows"]) == 6
    assert failure["decision_finished"] and not failure["retry_authorized"]


def test_initialization_failure_preserves_partial(tmp_path, setup):
    def runtime(*args):
        raise RuntimeError("model failed")
    with pytest.raises(RuntimeError, match="model failed"):
        d.run(tmp_path / "run", setup, digest(setup), d.AUTH, runtime=runtime)
    failure = json.loads((tmp_path / "run" / "failure.json").read_text())
    assert failure["stage"] == "runtime" and not failure["roster_returned"]
    assert not (tmp_path / "run" / "end.json").exists()


@pytest.mark.parametrize("change", ["pin", "authorization", "uuid"])
def test_admission_fails_before_output(tmp_path, setup, monkeypatch, change):
    p = setup
    pin, auth = digest(p), d.AUTH
    if change == "pin":
        pin = "0" * 64
    if change == "authorization":
        auth = "no"
    if change == "uuid":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(Exception):
        d.run(tmp_path / "run", p, pin, auth, runtime=lambda *a: None)
    assert not (tmp_path / "run").exists()


def test_inventory_rejects_symlink(tmp_path):
    (tmp_path / "a").symlink_to("/dev/null")
    with pytest.raises(Exception):
        d.inventory(tmp_path)


def test_real_file_pin_and_tamper(monkeypatch):
    p = plan()
    p["files"] = {k: d.file_sha(v) for k, v in {
        "driver": d.__file__, "adapter": d.adapter.__file__,
        "decision_journal": d.journal.__file__, "native_transport": d.transport.__file__}.items()}
    p["native_limits"] = {k: 0 for k in d.transport.base.LIMITS}
    monkeypatch.setattr(d.adapter, "checked_sources", lambda *a: ("contract", "prepared", "source"))
    assert d.validate(p)[0] == "contract"
    p["files"]["adapter"] = "0" * 64
    with pytest.raises(Exception, match="file pin"):
        d.validate(p)
