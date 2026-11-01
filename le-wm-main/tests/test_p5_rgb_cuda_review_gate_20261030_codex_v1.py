import sys

import pytest

from experiments.spectral_reachability import p5_rgb_cuda_review_gate_20261030_codex_v1 as gate


def event_bytes(event):
    body = dict(event)
    body["hash"] = gate.abi.bridge.digest(gate.abi.bridge.canonical(event))
    return gate.abi.bridge.canonical(body)


def start_event():
    return {"seq": 0, "prev": "0" * 64, "kind": "start",
            "prepared_sha256": gate.runner.PREP_SHA, "max_seconds": 30,
            "runner_source_sha256": gate.PINS[
                "experiments/spectral_reachability/p5_rgb_cuda_runner_20261030_codex_v1.py"],
            "backend": {"device": "cuda:0", "device_name": "fake", "torch_version": "fake",
                        "cuda_version": "fake", "strict_load": True, "frozen": True, "eval": True,
                        "checkpoint_sha256": gate.abi.bridge.FILES["checkpoint"][1]}}


def test_real_offline_preflight():
    report = gate.preflight()
    assert report["condition_inputs"]["off"]["branches"] == 82
    assert report["condition_inputs"]["on"]["branches"] == 82
    assert report["actual_neural_calls"] == report["actual_cuda_calls"] == 0
    assert "torch" not in sys.modules


def test_start_only_crash_preserves_all_slots():
    data = event_bytes(start_event())
    result = gate.strict_review(gate.runner.load_prepared(), data, gate.abi.bridge.digest(data))
    assert result["status"] == "partial_journal_verified"
    assert result["strict_envelope_verified"]
    assert len(result["rows"]) == 164
    assert result["condition_counts"]["on"]["complete"] == 0


def test_reject_rechained_extra_fields():
    event = start_event()
    event["future_label"] = "not permitted"
    data = event_bytes(event)
    prepared = gate.runner.load_prepared()
    # Demonstrates the earlier consumer accepts extra, internally rehashed fields.
    assert gate.consumer.consume(prepared, data)["status"] == "partial_journal_verified"
    with pytest.raises(ValueError, match="event field"):
        gate.strict_review(prepared, data, gate.abi.bridge.digest(data))


def test_reject_backend_extra_and_wrong_external_hash():
    event = start_event()
    event["backend"]["extra"] = "not permitted"
    data = event_bytes(event)
    prepared = gate.runner.load_prepared()
    with pytest.raises(ValueError, match="start envelope"):
        gate.strict_review(prepared, data, gate.abi.bridge.digest(data))
    with pytest.raises(ValueError, match="external journal SHA"):
        gate.strict_review(prepared, data, "0" * 64)


def test_reject_noncanonical_encoding():
    data = event_bytes(start_event()).replace(b'":', b'": ', 1)
    with pytest.raises(ValueError, match="canonical"):
        gate.strict_review(gate.runner.load_prepared(), data, gate.abi.bridge.digest(data))


def test_exclusive_write(tmp_path):
    path = tmp_path / "report.json"
    gate.write_new(path, {"test_only": True})
    with pytest.raises(FileExistsError):
        gate.write_new(path, {"test_only": False})
