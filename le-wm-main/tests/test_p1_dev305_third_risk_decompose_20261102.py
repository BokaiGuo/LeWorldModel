import pytest

from experiments.spectral_reachability import p1_dev305_third_risk_decompose_saved_20261102 as saved


def test_real_rgb_projection_fails_for_known_counterexample():
    result = saved.analyze()
    assert len(result["rows"]) == 6
    row = next(row for row in result["rows"] if row["arm"] == "zero__risk_selected")
    assert row["model_projection_visible"]
    assert row["real_latent_projection_visible"]
    assert not row["actual_visible"]
    assert row["actual_object_box"][0] == 6
    assert row["real_latent_object_box"][0] == pytest.approx(10.71457, abs=1e-4)
    assert not result["p1_admission"]
    assert not result["independent_confirmation"]
