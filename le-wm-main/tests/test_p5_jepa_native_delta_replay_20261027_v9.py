"""Non-neural preflight and algebra, not CUDA efficacy evidence."""
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import p5_jepa_native_delta_replay_20261027_v9 as subject

ROOT = Path(__file__).resolve().parents[1]


def test_fixed_ridge_delta():
    j = np.tile(np.eye(10), (2, 1))
    residual = np.arange(20, dtype=float)
    delta = subject.delta_from_arrays(j, residual)
    np.testing.assert_allclose((j.T @ j / 20 + .01 * np.eye(10)) @ delta, j.T @ residual / 20)


def test_zero_columns_retained():
    np.testing.assert_array_equal(subject.delta_from_arrays(np.zeros((20, 10)), np.ones(20)), np.zeros(10))


def test_nonfinite_rejected():
    with pytest.raises(ValueError):
        subject.delta_from_arrays(np.full((20, 10), np.nan), np.zeros(20))


def test_real_preparation_without_models():
    result = subject.prepare(ROOT)
    assert result["status"] == "prepared_non_neural"
    assert result["protocol"]["counts"] == dict(official_encoder=1, proprio_encoder=1, predictor=3)
    assert not result["protocol"]["selection"]
    assert not result["protocol"]["physical_execution"]
    assert not result["protocol"]["P5_admission"]
    assert len(result["delta_normalized"]) == 10


def test_manifest_drift_rejected_before_cuda(monkeypatch):
    monkeypatch.setattr(subject, "prepare", lambda root: {"correct": True})
    with pytest.raises(ValueError, match="manifest differs"):
        subject.run_cuda(ROOT, {"correct": False})
