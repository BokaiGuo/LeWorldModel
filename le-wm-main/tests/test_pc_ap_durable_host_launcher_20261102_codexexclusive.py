import hashlib
import json

import h5py
import numpy as np
import pytest

from experiments.spectral_reachability import pc_ap_durable_host_launcher_20261102_codexexclusive as host


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    policy = tmp_path / "policy"
    policy.mkdir()
    (policy / "weights.pt").write_bytes(b"weights")
    dataset = tmp_path / "data.h5"
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"source": {"dataset": str(dataset)}}))
    frames = np.arange(93 * 2 * 2 * 3, dtype=np.uint8).reshape(93, 2, 2, 3)
    with h5py.File(dataset, "w") as f:
        f["ep_offset"] = [0, 0, 31, 62]
        f["ep_len"] = [31, 31, 31, 31]
        f["ep_idx"] = np.repeat([0, 2, 3], 31)
        f["pixels"] = frames
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"weights_sha256": digest(b"weights"),
                                  "stats_sha256": host.sha_file(stats),
                                  "memory_rgb_sha256": {str(k): digest(frames[i*31:(i+1)*31].tobytes())
                                                        for i, k in enumerate((0, 2, 3))}}))
    monkeypatch.setattr(host, "REPORT_SHA", host.sha_file(report))
    monkeypatch.setattr(host, "SOURCE_SHA", {})
    return policy, stats, dataset, report


def test_asset_authentication_and_tampered_memory(inputs):
    policy, stats, dataset, report = inputs
    _, episodes, info = host.check_assets(policy, stats, dataset, report=report)
    assert list(episodes) == ["0", "2", "3"]
    assert info["2"]["offset"] == 31
    with h5py.File(dataset, "r+") as f:
        f["pixels"][31, 0, 0, 0] ^= 1
    with pytest.raises(ValueError, match="pixels SHA"):
        host.check_assets(policy, stats, dataset, report=report)


def test_wrong_episode_and_checkpoint_rejected(inputs):
    policy, stats, dataset, report = inputs
    with h5py.File(dataset, "r+") as f:
        f["ep_idx"][0] = 2
    with pytest.raises(ValueError, match="episode provenance"):
        host.check_assets(policy, stats, dataset, report=report)
    (policy / "weights.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA mismatch"):
        host.check_assets(policy, stats, dataset, report=report)


def test_dataset_source_and_report_binding(inputs, tmp_path, monkeypatch):
    policy, stats, dataset, report = inputs
    copied = tmp_path / "other.h5"
    copied.write_bytes(dataset.read_bytes())
    with pytest.raises(ValueError, match="dataset differs"):
        host.check_assets(policy, stats, copied, report=report)
    monkeypatch.setattr(host, "REPORT_SHA", "0" * 64)
    with pytest.raises(ValueError, match="SHA mismatch"):
        host.check_assets(policy, stats, dataset, report=report)


def test_reference_requires_complete_paired_identity():
    row = {"seed": 1, "status": "complete", "root_rgb_sha256": "r", "goal_rgb_sha256": "g"}
    assert host._roots({"rows": [row, row]}, 1) == ("r", "g")
    with pytest.raises(ValueError, match="incomplete"):
        host._roots({"rows": [row, dict(row, goal_rgb_sha256="other")]}, 1)


def test_busy_lock_prevents_output_and_work(monkeypatch, tmp_path):
    import contextlib
    from experiments.spectral_reachability import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as lock

    @contextlib.contextmanager
    def busy(_):
        raise BlockingIOError("busy")
        yield

    monkeypatch.setattr(host, "check_assets", lambda *a: None)
    monkeypatch.setattr(lock, "gpu_lock", busy)
    output = tmp_path / "output"
    monkeypatch.setattr(host, "OUTPUT", output)
    with pytest.raises(BlockingIOError):
        host.execute("policy", "stats", "dataset")
    assert not output.exists()


def test_predictor_meter_records_actual_failure_and_restores():
    import types

    class Predictor:
        def __init__(self):
            self.predicted_blocks = 0
            self.prediction_batches = 0

        def __call__(self, actions):
            self.predicted_blocks += len(actions)
            self.prediction_batches += 1
            raise RuntimeError("model failed")

    module = types.SimpleNamespace(ReacherCudaPredictor=Predictor)
    events = []
    journal = types.SimpleNamespace(emit=events.append)
    with host.predictor_meter(module, journal):
        with pytest.raises(RuntimeError, match="model failed"):
            module.ReacherCudaPredictor()([1, 2, 3])
    assert module.ReacherCudaPredictor is Predictor
    assert events[-1] == {"kind": "predictor_call_failed", "blocks": 3,
                          "batches": 1, "error_type": "RuntimeError"}
