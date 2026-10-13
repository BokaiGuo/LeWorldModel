import numpy as np

from experiments.run_pusht_observability_decoder_ablation import bootstrap


def test_bootstrap_is_finite_and_tracks_mean():
    result = bootstrap(np.asarray([1.0, 2.0, 3.0, 4.0]), seed=4, draws=200)
    assert result["estimate"] == 2.5
    assert np.isfinite(result["ci95_low"])
    assert np.isfinite(result["ci95_high"])

