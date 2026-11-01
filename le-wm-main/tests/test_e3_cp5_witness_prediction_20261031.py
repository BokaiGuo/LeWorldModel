import numpy as np
import pytest

from experiments.spectral_reachability import e3_cp5_witness_prediction_20261031 as experiment


def test_collect_preserves_both_roots_and_valid_action_abi():
    rows = experiment.collect()
    assert [row["root"] for row in rows] == [0, 1]
    for row in rows:
        assert row["states"].shape == (3, 192)
        assert row["history"].shape == (2, 10)
        assert row["witness"].shape == (5, 2)
        assert np.all(row["std"] > 0)
        token = ((row["witness"] - row["mean"]) / row["std"]).reshape(1, 10)
        assert np.concatenate((row["history"], token)).shape == (3, 10)


def test_rejects_modified_target_source(monkeypatch):
    original = experiment.base.sha

    def altered(path):
        if path == experiment.TARGET:
            return "0" * 64
        return original(path)

    monkeypatch.setattr(experiment.base, "sha", altered)
    with pytest.raises(ValueError, match="input identity"):
        experiment.collect()


def test_rejects_bad_dimensions_and_nonfinite():
    with pytest.raises(ValueError):
        experiment.array([float("nan")] * 2, (2,), "bad")
    with pytest.raises(ValueError):
        experiment.array([0.0] * 3, (2,), "bad")
