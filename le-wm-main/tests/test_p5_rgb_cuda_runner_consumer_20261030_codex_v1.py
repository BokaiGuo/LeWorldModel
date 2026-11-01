import sys

import numpy as np
import pytest

from experiments.spectral_reachability import p5_rgb_cuda_runner_20261030_codex_v1 as runner
from experiments.spectral_reachability import p5_rgb_cuda_consumer_20261030_codex_v1 as consumer


class FakeBackend:
    """Data-only test double; never imports or executes a neural model."""
    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def metadata(self):
        return {"device": "cuda:0", "device_name": "synthetic test double",
                "torch_version": "not executed", "cuda_version": "not executed",
                "strict_load": True, "frozen": True, "eval": True,
                "checkpoint_sha256": runner.abi.bridge.FILES["checkpoint"][1]}

    def result(self):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("injected failure")
        return np.full((1, 6), self.calls, dtype=np.float32)

    def encode(self, values):
        return self.result()

    def step(self, values, index):
        return self.result()


def simulated(tmp_path, monkeypatch, fail_at=None):
    prepared = runner.load_prepared()
    monkeypatch.setattr(runner.abi, "materialize", lambda row: {})
    journal = runner.Journal(tmp_path / "run")
    backend = FakeBackend(fail_at)
    try:
        runner.run_protocol(prepared, backend, journal, 3600)
    finally:
        journal.close()
    return prepared, (tmp_path / "run" / "events.jsonl").read_bytes(), backend.calls


def test_complete_and_separate_conditions(tmp_path, monkeypatch):
    p, data, calls = simulated(tmp_path, monkeypatch)
    receipt = consumer.consume(p, data)
    assert calls == 984
    assert receipt["status"] == "complete_journal_verified"
    assert receipt["condition_counts"] == {
        "off": {"denominator": 82, "complete": 82},
        "on": {"denominator": 82, "complete": 82}}
    assert len(receipt["rows"]) == 164
    assert not receipt["P5_admission"]
    assert "torch" not in sys.modules


def test_failed_transition_keeps_full_denominator(tmp_path, monkeypatch):
    p, data, calls = simulated(tmp_path, monkeypatch, fail_at=3)
    result = consumer.consume(p, data)
    assert calls == 3
    assert result["status"] == "partial_journal_verified"
    assert result["rows"][0]["steps"][1]["status"] == "failed"
    assert result["rows"][163]["steps"][4]["status"] == "unattempted"
    assert result["condition_counts"]["off"]["denominator"] == 82


def test_crash_with_attempt_remains_unreturned(tmp_path, monkeypatch):
    p, data, _ = simulated(tmp_path, monkeypatch)
    lines = data.splitlines(keepends=True)
    result = consumer.consume(p, b"".join(lines[:2]))
    assert result["end_status"] is None
    assert result["rows"][0]["encode"]["status"] == "attempted_unreturned"


def test_tamper_and_truncation_rejected(tmp_path, monkeypatch):
    p, data, _ = simulated(tmp_path, monkeypatch, fail_at=2)
    with pytest.raises(ValueError):
        consumer.consume(p, data.replace(b"predict_next", b"predict_nexx", 1))
    with pytest.raises(ValueError):
        consumer.consume(p, data[:-1])


def test_input_preparation_pinned_and_no_torch():
    runner.verify_sources()
    row = runner.load_prepared()["rows"][0]
    values = runner.abi.materialize(row)
    assert set(values) == {"previous_image", "image", "actions"}
    assert "torch" not in sys.modules
