import copy

import pytest

from experiments.spectral_reachability import codex_p1_cp5_midframe_replay_20261030_main as subject


def test_fixed_midpoint_identity_and_pixels():
    _, entries = subject.prior.inputs()
    assert len(entries) == 48
    for entry in entries:
        rgb, _, frame = subject.midpoint(entry)
        assert rgb.shape == (224, 224, 3)
        assert frame["step"] == 5
        assert frame["condition"] == entry["identity"]["condition"]


def test_rejects_wrong_condition():
    _, entries = subject.prior.inputs()
    entry = copy.deepcopy(entries[0])
    entry["identity"]["condition"] = "on" if entry["identity"]["condition"] == "off" else "off"
    with pytest.raises(ValueError, match="midpoint"):
        subject.midpoint(entry)
