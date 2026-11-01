import numpy as np
import pytest
from experiments.spectral_reachability.p1_joint_suffix_scale_saved_20261029 import costs


def test_actual_plus_vectors_are_scored():
    row = {'zero': np.zeros(192).tolist(), 'scales': [
        {'scale': .1, 'plus': np.ones(192).tolist()},
        {'scale': 1., 'plus': (2*np.ones(192)).tolist()}]}
    result = costs(np.ones(192), row)
    assert result == {'zero': 1., '0.1': 0., '1.0': 1.}


def test_nonfinite_rejected():
    with pytest.raises(ValueError):
        costs(np.full(192, np.nan), {'zero': np.zeros(192).tolist(), 'scales': []})
