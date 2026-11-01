"""Source and binding checks for the known-root CUDA target experiment."""
import pytest

from experiments.spectral_reachability import e3_cp5_target_cuda_20261031 as target


def test_collects_two_witnessed_targets():
    entries, sources = target.collect()
    assert [entry["root"] for entry in entries] == [0, 1]
    assert all(entry["pixels"].shape == (224, 224, 3) for entry in entries)
    assert all(entry["existing"].shape == (192,) for entry in entries)
    assert all(len(entry["witness"]) == 5 for entry in entries)
    assert len(sources) == 6


def test_rejects_unverified_resolver(monkeypatch):
    original = target.base.sha

    def altered(path):
        if path == target.qualification.INPUT:
            return "0" * 64
        return original(path)

    monkeypatch.setattr(target.base, "sha", altered)
    with pytest.raises(ValueError, match="resolver source"):
        target.collect()
