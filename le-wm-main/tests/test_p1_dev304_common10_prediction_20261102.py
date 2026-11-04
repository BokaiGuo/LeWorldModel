import pytest

from experiments.spectral_reachability import p1_dev304_common10_prediction_saved_20261102 as saved


def test_saved_prediction_panel():
    result = saved.analyze()
    assert len(result["rows"]) == len(result["paired_rankings"]) == 6
    assert not result["p1_admission"] and not result["independent_confirmation"]


def test_reject_bad_latent_shape():
    with pytest.raises(ValueError):
        saved.mse([0.] * 191, [0.] * 192)
