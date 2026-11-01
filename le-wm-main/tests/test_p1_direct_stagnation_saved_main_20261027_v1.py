import numpy as np
import pytest
from experiments.spectral_reachability import p1_direct_stagnation_saved_main_20261027_v1 as m


def test_product_ball_projection_not_global_ball():
    a = np.zeros((2, 10)); a[0, 0] = 3.; a[1, 0] = .2
    p = m.project(a, 1.)
    np.testing.assert_array_equal(p[:, 0], [1., .2])


def test_stationary_boundary_normal_has_zero_mapping():
    x = np.zeros((2, 10)); x[:, 0] = 1.
    g = -x
    np.testing.assert_array_equal(x - m.project(x-g, 1.), 0.)


def test_tangent_gradient_is_not_stationary():
    x = np.zeros((2, 10)); x[:, 0] = 1.
    g = np.zeros_like(x); g[:, 1] = .3
    assert np.linalg.norm(x - m.project(x-g, 1.)) > .1


@pytest.mark.parametrize('radius', [0., -1., float('nan')])
def test_bad_radius(radius):
    with pytest.raises(ValueError):
        m.project(np.zeros((2, 10)), radius)


def test_actual_four_trajectories():
    report = m.read(m.RUN / 'report.json')
    tasks = {m.read(m.RUN / f'task_{i}.json.gz')['root_sha256']:
             m.read(m.RUN / f'task_{i}.json.gz')['task'] for i in range(2)}
    for arm in report['direct']:
        result = m.analyze(tasks[arm['root_sha256']], m.read(arm['report_path']))
        assert len(result['rows']) == 8 and result['stationarity_threshold'] is None
        assert result['B8_cost'] <= result['B4_cost']
