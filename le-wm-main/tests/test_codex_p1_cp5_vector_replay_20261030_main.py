import copy

import pytest

from experiments.spectral_reachability import codex_p1_cp5_vector_replay_20261030_main as run


def test_fixed_replay_inputs_bind_complete_panel():
    bundle, entries = run.inputs()
    assert len(bundle["rows"]) == 24
    assert len(entries) == 48
    assert all(entry["actual_latent"].shape == (192,) for entry in entries)


def test_rgb_identity_mismatch_fails(monkeypatch):
    original_read = run.gap.read

    def altered(path, expected=None):
        value, digest = original_read(path, expected)
        if path == run.RGB:
            value = copy.deepcopy(value)
            value["rows"][0]["event_sha256"] = "0" * 64
        return value, digest

    monkeypatch.setattr(run.gap, "read", altered)
    with pytest.raises(ValueError, match="RGB result identity mismatch"):
        run.inputs()
