import json

import pytest

from experiments.spectral_reachability import p3_h3_model_physical_direction_20261031 as audit


def inputs():
    return [json.loads((audit.ROOT / "results" / audit.SOURCES[key][0]).read_text())
            for key in ("prepare", "model", "physical")]


def test_real_roster_and_censoring():
    result = audit.analyze(*inputs())
    assert result["unique_action_pairs"] == 18
    assert result["complete_pairs"] == 13
    assert result["direction_counts_model_then_physical"]["native_distance"]["lower/higher"] == 8
    assert result["P3_admission"] is False


def test_wrong_suffix_is_rejected():
    prepare, model, physical = inputs()
    prepare["plans"][0]["commands"][-1][0] += 0.01
    with pytest.raises(ValueError, match="exact suffix"):
        audit.analyze(prepare, model, physical)


def test_bad_prefix_replay_is_rejected():
    prepare, model, physical = inputs()
    physical["groups"][0]["records"][0]["step15_exact"] = False
    with pytest.raises(ValueError, match="shared prefix"):
        audit.analyze(prepare, model, physical)
