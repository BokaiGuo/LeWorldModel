import numpy as np

from experiments.run_pusht_temporal_visual_adapter import bootstrap


def test_temporal_adapter_bootstrap_is_finite():
    result = bootstrap(np.asarray([0.0, 1.0, 2.0, 3.0]), seed=3, draws=100)
    assert np.isfinite(result["estimate"])
    assert np.isfinite(result["ci95_low"])
    assert np.isfinite(result["ci95_high"])
