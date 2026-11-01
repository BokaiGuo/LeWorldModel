import pytest
from experiments.spectral_reachability import main_p3_goal_bank_curvature_20261029_v1 as m


def test_linear_model_exact():
    r = m.decompose([1.], [0.], [[2.]], [.1], [.2])
    assert r['first_order'] == pytest.approx(-.4)
    assert r['quadratic'] == pytest.approx(.04)
    assert r['delta'] == pytest.approx(-.36)
    assert r['nonlinear_total'] == 0


def test_curvature_can_reverse_descent_without_wrong_J():
    # f(u)=u-20u^2 has exact J(0)=1, but u=.1 moves away from goal.
    r = m.decompose([1.], [0.], [[1.]], [.1], [-.1])
    assert r['first_order'] < 0 < r['delta']
    assert r['nonlinear_total'] > 0


def test_zero_action_and_residual():
    r = m.decompose([1., 2.], [1., 2.], [[1.], [2.]], [0.], [1., 2.])
    assert all(v == 0 for v in r.values())


@pytest.mark.parametrize('bad', [True, float('nan'), float('inf')])
def test_nonfinite_and_bool_refused(bad):
    with pytest.raises(ValueError):
        m.decompose([bad], [0.], [[1.]], [1.], [0.])


def test_dimension_mismatch():
    with pytest.raises(ValueError):
        m.decompose([1.], [0.], [[1., 2.]], [1.], [0.])
