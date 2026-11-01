from copy import deepcopy

import numpy as np
import pytest

from experiments.spectral_reachability import e3_cp5_four_arm_root1_20261031 as experiment


def loaded():
    experiment.configure()
    prepared = experiment.build()
    data = experiment.runner.p3.source_data(prepared["source"])
    return prepared, data


def test_goal_rebinding_matches_fixed_source_and_preserves_original():
    prepared, data = loaded()
    original = deepcopy(prepared["source"]["tasks"][1])
    binding = experiment.rebind(prepared, data)
    ref = data["references"][1]
    assert binding["initial_success"] is False
    assert binding["goal_pixel_sha256"] == experiment.inputs()[1]["pixel_sha256"]
    assert experiment.qualification.native(ref["goal"], ref["states"][-1])["initial_success"] is False
    assert prepared["source"]["tasks"][1]["goal"] != original["goal"]
    assert prepared["source"]["tasks"][1]["root_context"] == original["root_context"]
    assert np.array_equal(ref["goal_rgb"], experiment.inputs()[0]["pixels"])


def test_wrong_root_is_rejected():
    prepared, data = loaded()
    data["references"][1]["states"][-1][0] += 1
    with pytest.raises(ValueError, match="same physical starting root"):
        experiment.rebind(prepared, data)
