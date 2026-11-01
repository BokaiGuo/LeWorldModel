import copy

import pytest

from experiments.spectral_reachability import p1_uphill_multistart_rgb_20261030_main as rgb


def test_all_uncensored_endpoints_are_bound():
    entries, sources = rgb.inputs()
    assert len(entries) == 19
    assert len({row["index"] for row in entries}) == 19
    assert all(rgb.base.sha(path) == digest for path, digest in sources.items())


def test_rejects_mismatched_pair_identity(monkeypatch):
    original = rgb.paired.read

    def modified(path, digest):
        data = original(path, digest)
        if path == rgb.PAIR:
            data = copy.deepcopy(data)
            data["rows"][0]["role"] = "invalid"
        return data

    monkeypatch.setattr(rgb.paired, "read", modified)
    with pytest.raises(ValueError, match="row identity"):
        rgb.inputs()
