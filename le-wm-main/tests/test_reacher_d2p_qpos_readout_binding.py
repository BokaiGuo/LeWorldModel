import hashlib
import json

import numpy as np
import pytest
import torch

from experiments.spectral_reachability.reacher_d2p_qpos_readout import (
    PREDICTION_READOUT_TASK,
    READOUT_TASK,
    qpos_readout_binding,
    with_qpos_readout,
)


def _artifact(path, task):
    np.savez_compressed(
        path,
        feature_mean=np.zeros(2, dtype=np.float32),
        feature_scale=np.ones(2, dtype=np.float32),
        coefficient=np.ones((3, 4), dtype=np.float32),
        task=np.asarray([task]),
    )


def _report(path, *, schema, policy_hash, adapter_hash):
    path.parent.joinpath("report.json").write_text(json.dumps({
        "schema": schema,
        "status": "completed",
        "validation": {"gate_passed": True},
        "inputs": {"weights_sha256": policy_hash, "adapter_sha256": adapter_hash},
        "artifact": {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
    }), encoding="utf-8")


def test_p1_qpos_binding_requires_a_complete_readout_pair(tmp_path):
    encoder, predicted = tmp_path / "encoder.npz", tmp_path / "predicted.npz"
    _artifact(encoder, READOUT_TASK)
    _artifact(predicted, PREDICTION_READOUT_TASK)

    assert qpos_readout_binding(None, None) == {"enabled": False}
    with pytest.raises(ValueError, match="both encoder"):
        qpos_readout_binding(encoder, None)

    binding = qpos_readout_binding(encoder, predicted)
    assert binding["enabled"]
    assert binding["encoder_artifact_sha256"] == hashlib.sha256(encoder.read_bytes()).hexdigest()
    assert binding["prediction_artifact_sha256"] == hashlib.sha256(predicted.read_bytes()).hexdigest()


def test_p1_qpos_binding_wraps_without_altering_frozen_predictor_calls(tmp_path):
    encoder, predicted = tmp_path / "encoder.npz", tmp_path / "predicted.npz"
    _artifact(encoder, READOUT_TASK)
    _artifact(predicted, PREDICTION_READOUT_TASK)

    class Base(torch.nn.Module):
        action_encoder = staticmethod(lambda action: action)

        def encode(self, info):
            return {"emb": info["pixels"]}

        def predict(self, embeddings, actions):
            return embeddings + actions

    binding = qpos_readout_binding(encoder, predicted)
    model = with_qpos_readout(Base(), binding)
    assert torch.equal(model.predict(torch.ones(1, 1, 2), torch.ones(1, 1, 2)), torch.full((1, 1, 2), 2.0))


def test_p1_qpos_binding_rejects_readout_calibrated_for_a_different_checkpoint(tmp_path):
    encoder_dir, predicted_dir = tmp_path / "encoder", tmp_path / "predicted"
    encoder_dir.mkdir(); predicted_dir.mkdir()
    encoder, predicted = encoder_dir / "qpos_readout.npz", predicted_dir / "predicted_qpos_readout.npz"
    _artifact(encoder, READOUT_TASK); _artifact(predicted, PREDICTION_READOUT_TASK)
    _report(encoder, schema="reacher_d2p_frozen_latent_qpos_readout.v1", policy_hash="policy-a", adapter_hash="adapter")
    _report(predicted, schema="reacher_d2p_frozen_rollout_qpos_readout.v1", policy_hash="policy-a", adapter_hash="adapter")

    with pytest.raises(ValueError, match="provenance"):
        qpos_readout_binding(encoder, predicted, policy_weights_sha256="policy-b", adapter_sha256="adapter")
    binding = qpos_readout_binding(encoder, predicted, policy_weights_sha256="policy-a", adapter_sha256="adapter")
    assert set(binding["calibration_reports"]) == {READOUT_TASK, PREDICTION_READOUT_TASK}
