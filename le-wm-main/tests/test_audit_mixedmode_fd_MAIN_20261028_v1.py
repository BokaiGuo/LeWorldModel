import numpy as np
import pytest
from experiments.spectral_reachability import audit_mixedmode_fd_MAIN_20261028_v1 as m


@pytest.mark.parametrize('coordinate', range(10))
@pytest.mark.parametrize('epsilon', m.EPS)
def test_full_vector_difference(coordinate, epsilon):
    j = np.arange(1920, dtype=float).reshape(192, 10)/1000
    h = float(np.float32(epsilon))
    out = m.compare(j, coordinate, epsilon, -h*j[:, coordinate], h*j[:, coordinate])
    assert out['relative_l2'] < 1e-14


def test_transposed_derivative_rejected():
    with pytest.raises(ValueError):
        m.compare(np.zeros((10, 192)), 0, .01, np.zeros(192), np.zeros(192))


def test_wrong_scale_is_detected_not_relabelled_success():
    out = m.compare(np.ones((192, 10)), 0, .01, -np.ones(192), np.ones(192))
    assert out['relative_l2'] > .98
