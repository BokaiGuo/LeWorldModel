import numpy as np
import pytest

from experiments.spectral_reachability import main_selector_gap_decomposition_20261028_v1 as m


def sample(curvature=0., stage=0., rounding=False):
    j = np.zeros((192, 10)); j[0, 0] = 2.
    u = np.zeros(10); u[0] = .123456789 if rounding else .25
    goal = np.zeros(192); goal[0] = 1.
    nominal = np.zeros(192)
    terminal = nominal+j@u.astype(np.float32).astype(float)
    terminal[0] += curvature
    prefix = goal.copy(); prefix[1] += stage
    linear = float(np.sum((nominal+j@u-goal)**2)/192)
    ridge = float(.002*(u@u)/192)
    c = dict(index=0, jacobian=j.tolist(), nominal_terminal=nominal.tolist(),
        predicted_terminal=terminal.tolist(), predicted_prefix=prefix.tolist(), suffix_params=u.tolist(),
        suffix_commands=np.tanh(u.astype(np.float32)).reshape(5,2).tolist(),
        suffix_solve=dict(linear_terminal_cost=linear, ridge_penalty=ridge),
        own_reach_value=linear+ridge, stage_cost=stage**2/192,
        objective=stage**2/192+linear+ridge,
        nonlinear_terminal_cost=float(np.sum((terminal-goal)**2)/192))
    return c, goal, np.eye(192), goal


@pytest.mark.parametrize('curvature', [0., .1, -.1, 2.])
@pytest.mark.parametrize('rounding', [False, True])
def test_exact_signed_decomposition(curvature, rounding):
    out = m.decompose(*sample(curvature, .2, rounding))
    assert out['objective']-out['terminal_cost']==pytest.approx(out['selector_gap'], abs=1e-15)
    if not rounding:
        assert out['rounding_cost_change']==0.


def test_cross_term_cancellation_is_preserved():
    out = m.decompose(*sample(.1))
    assert out['nonlinear_cross']<0<out['nonlinear_square']
    assert out['nonlinear_cost_change']<0


@pytest.mark.parametrize('field', ['objective', 'nonlinear_terminal_cost', 'stage_cost', 'own_reach_value'])
def test_reject_stale_score(field):
    c, g, q, r = sample()
    c[field] += .01
    with pytest.raises(ValueError):
        m.decompose(c,g,q,r)


def test_reject_raw_mismatch():
    c,g,q,r=sample(); c['suffix_commands'][0][0] += .01
    with pytest.raises(ValueError, match='literal'):
        m.decompose(c,g,q,r)


def test_pairwise_identity():
    a=m.decompose(*sample(.1,.4)); b=m.decompose(*sample(-.2,.1,True)); b['index']=1
    pair=m.contrast(a,b)
    assert pair['objective_delta']-pair['terminal_delta']==pytest.approx(sum(pair['signed_gap_contributions'].values()), abs=1e-15)


def test_nonfinite_rejected():
    c,g,q,r=sample(); c['predicted_terminal'][0]=float('nan')
    with pytest.raises(ValueError, match='finite'):
        m.decompose(c,g,q,r)
