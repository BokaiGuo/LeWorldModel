import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p5_rgb_instrumented_supervision_20261030_codex_v1 as mod

GPU = "GPU-12345678-1234-1234-1234-123456789abc"


class FakeBackend:
    model = object()
    torch = object()

    def __init__(self, fail_on=None):
        self.calls = 0
        self.fail_on = fail_on

    def metadata(self):
        return {"device": "cuda:0", "device_name": "synthetic only", "torch_version": "fake",
                "cuda_version": "fake", "strict_load": True, "frozen": True, "eval": True,
                "checkpoint_sha256": mod.abi.bridge.FILES["checkpoint"][1]}

    def value(self):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("synthetic failure")
        return np.zeros((1, 6), dtype=np.float32)

    def encode(self, values):
        return self.value()

    def step(self, values, index):
        return self.value()


@pytest.fixture
def setup_data(monkeypatch):
    base = mod.preflight()
    state = copy.deepcopy(base["state"])
    for record in state["parameters"].values():
        record["device"] = "cuda:0"
    for record in state["normalization"].values():
        record["device"] = "cuda:0"
    monkeypatch.setattr(mod.abi, "materialize", lambda row: {})
    return base, state


def simulate(directory, base, state, fail_on=None, change_after=False):
    journal = mod.runner.Journal(directory)
    backend = FakeBackend(fail_on)
    snapshots = []
    def capture(model, torch):
        value = copy.deepcopy(state)
        if change_after and snapshots:
            value["modules"][0]["training"] = True
        snapshots.append(value)
        return value
    wrapped = mod.InstrumentedBackend(backend, journal, directory, base, capture=capture)
    try:
        status = mod.runner.run_protocol(mod.runner.load_prepared(), wrapped, journal, 3600)
    finally:
        journal.close()
    return status, backend, snapshots


def test_full_wiring_both_anchors_and_review(tmp_path, setup_data):
    base, state = setup_data
    directory = tmp_path / "run"
    status, backend, snapshots = simulate(directory, base, state)
    assert status == "complete" and backend.calls == 984 and len(snapshots) == 2
    review = mod.review_run(directory, base, mod.source_sha())
    assert review["condition_counts"]["off"]["complete"] == 82
    assert review["condition_counts"]["on"]["complete"] == 82
    first = json.loads((directory / "state_before.json").read_bytes())
    last = json.loads((directory / "state_after.json").read_bytes())
    assert first["anchor_seq"] == 1 and last["anchor_seq"] == 1967


@pytest.mark.parametrize("fail_on,change_after", [(3, False), (None, True)])
def test_failure_never_creates_accepted_complete(tmp_path, setup_data, fail_on, change_after):
    base, state = setup_data
    directory = tmp_path / "run"
    status, backend, snapshots = simulate(directory, base, state, fail_on, change_after)
    assert status == "partial"
    assert (directory / "state_before.json").exists()
    assert not (directory / "state_after.json").exists()
    with pytest.raises(ValueError):
        mod.review_run(directory, base, mod.source_sha())


def test_tampered_snapshot_anchor_rejected(tmp_path, setup_data):
    base, state = setup_data
    directory = tmp_path / "run"
    simulate(directory, base, state)
    path = directory / "state_after.json"
    record = json.loads(path.read_bytes())
    record["anchor_hash"] = "0" * 64
    path.write_bytes(mod.abi.bridge.canonical(record))
    with pytest.raises(ValueError, match="anchor"):
        mod.review_run(directory, base, mod.source_sha())


def test_launch_to_accept_integration_no_gpu(tmp_path, monkeypatch, setup_data):
    base, state = setup_data
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(mod.abi.bridge, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "preflight", lambda: base)
    monkeypatch.setattr(mod.sup, "query_gpu", lambda uuid: {
        "uuid": uuid, "total_mib": 2000, "free_mib": 1500,
        "compute_mode": "Default", "mig_mode": "Disabled"})
    calls = []
    def supervise(command, env, cwd, out, err, timeout, on_started):
        calls.append((command, env["CUDA_VISIBLE_DEVICES"], timeout))
        on_started(123)
        status, _, _ = simulate(tmp_path / "results" / (mod.NAME + "_run_TEST"), base, state)
        assert status == "complete"
        return {"status": "exited", "returncode": 0, "pid": 123, "elapsed_seconds": 1.0, "kill": None}
    monkeypatch.setattr(mod.sup, "supervise", supervise)
    args = SimpleNamespace(authorization=mod.AUTH, gpu_uuid=GPU, run_suffix="TEST",
                           min_free_mib=100, min_total_mib=2000, hard_seconds=60, runner_seconds=30)
    assert mod.launch(args) == 0
    assert len(calls) == 1 and calls[0][1] == GPU
    log = (tmp_path / "results" / (mod.NAME + "_supervisor_TEST") / "supervisor.jsonl").read_bytes()
    journal = (tmp_path / "results" / (mod.NAME + "_run_TEST") / "events.jsonl").read_bytes()
    verify = SimpleNamespace(run_suffix="TEST", supervisor_sha=mod.abi.bridge.digest(log),
                             journal_sha=mod.abi.bridge.digest(journal),
                             output_name=mod.NAME + "_accept_TEST.json")
    assert mod.accept(verify) == 0
    assert json.loads((tmp_path / "results" / verify.output_name).read_bytes())["P5_admission"] is False
    with pytest.raises(FileExistsError):
        mod.accept(verify)


def test_supervisor_timeout_partial_no_retry(tmp_path, monkeypatch, setup_data):
    base, _ = setup_data
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(mod.abi.bridge, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "preflight", lambda: base)
    monkeypatch.setattr(mod.sup, "query_gpu", lambda uuid: {
        "uuid": uuid, "total_mib": 2000, "free_mib": 1500,
        "compute_mode": "Default", "mig_mode": "Disabled"})
    calls = []
    def timeout(command, env, cwd, out, err, seconds, on_started):
        calls.append(1)
        on_started(123)
        return {"status": "timeout", "returncode": -9, "pid": 123, "elapsed_seconds": 1, "kill": "sigkill_sent"}
    monkeypatch.setattr(mod.sup, "supervise", timeout)
    args = SimpleNamespace(authorization=mod.AUTH, gpu_uuid=GPU, run_suffix="TEST",
                           min_free_mib=100, min_total_mib=2000, hard_seconds=60, runner_seconds=30)
    assert mod.launch(args) == 2 and len(calls) == 1
    log = (tmp_path / "results" / (mod.NAME + "_supervisor_TEST") / "supervisor.jsonl").read_bytes()
    assert json.loads(log.splitlines()[-1])["status"] == "partial"
