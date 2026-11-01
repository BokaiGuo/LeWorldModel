import copy
import json

import pytest

from experiments.spectral_reachability import p3_rgb_fallback_exploration_20261031 as run


@pytest.fixture(scope="module")
def inputs():
    return [json.loads(path.read_bytes()) for path in (run.GUARDED, run.BASE)]


def test_fixed_panel_uses_actual_old_branch_only_on_abstention(inputs):
    result = run.analyze(*inputs)
    assert len(result["rows"]) == 52
    assert result["summary"]["unique_pairs"] == 26
    assert result["summary"]["selected_origins"] == {
        "guarded_new_observed": 20, "cached_old_counterfactual": 6}
    for row in result["rows"]:
        if row["origin"] == "cached_old_counterfactual":
            assert row["endpoint"] == row["old_endpoint"]
            assert all(v == 0 for v in row["delta_vs_old"].values())
    assert not result["P3_admission"]


def test_rejects_identity_mismatch_and_missing_old_outcome(inputs):
    guarded, prior = copy.deepcopy(inputs)
    prior["rows"][0]["pair_key"] = "bad"
    with pytest.raises(ValueError):
        run.analyze(guarded, prior)
    guarded, prior = copy.deepcopy(inputs)
    prior["rows"][0]["old_endpoint"] = None
    with pytest.raises(ValueError):
        run.analyze(guarded, prior)
