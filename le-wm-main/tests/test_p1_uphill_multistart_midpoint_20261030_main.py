import copy

import pytest

from experiments.spectral_reachability import p1_uphill_multistart_midpoint_20261030_main as mid


def test_inputs_all_fixed_nonstopped_rows():
    entries, sources = mid.inputs()
    assert len(entries) == 19
    assert all(entry["midpoint"].shape == (224, 224, 3) for entry in entries)
    assert all(mid.base.sha(path) == digest for path, digest in sources.items())


def test_rejects_corrupt_terminal_identity(monkeypatch):
    original = mid.rgb.paired.read

    def altered(path, digest):
        result = original(path, digest)
        if path == mid.RGB:
            result = copy.deepcopy(result)
            result["rows"][0]["pixel_sha256"] = "invalid"
        return result

    monkeypatch.setattr(mid.rgb.paired, "read", altered)
    with pytest.raises(ValueError, match="terminal RGB identity"):
        mid.inputs()
