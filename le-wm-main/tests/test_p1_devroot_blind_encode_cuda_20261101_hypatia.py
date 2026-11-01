import pytest

from experiments.spectral_reachability import p1_devroot_blind_encode_cuda_20261101_hypatia as dev


def test_cpu_build_seals_exact_two_dev_packets_and_original_normalizer():
    plan = dev.build()
    assert [r["slot"] for r in plan["slots"]] == [0, 1]
    assert plan["normalization"] == dev.moving.normalization()
    assert plan["planner_authorized"] is False and plan["task_export"] is False
    assert plan["freshness"] == plan["training_disjoint"] == "unknown"


def test_payload_identity_drift_rejected(monkeypatch, tmp_path):
    path = tmp_path / "fake_blind_inputs.npz"
    path.write_bytes(b"not a blind payload")
    monkeypatch.setattr(dev, "SLOTS", [(path, "0" * 64), (path, "0" * 64)])
    with pytest.raises(ValueError):
        dev.build()
