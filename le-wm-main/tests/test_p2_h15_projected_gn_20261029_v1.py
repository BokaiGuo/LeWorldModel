import numpy as np
import pytest
from experiments.spectral_reachability import p2_h15_projected_gn_20261029_v1 as m


def test_fixed_budget_and_domain():
    calls = [0, 0]
    def evaluate(a):
        calls[0] += 1
        out = np.zeros(192)
        out[:30] = np.asarray(a).reshape(30)
        return out
    def linearize(z):
        calls[1] += 1
        value = np.zeros(192)
        value[:30] = m.raw(z).reshape(30)
        j = np.zeros((192, 30))
        j[:30] = np.diag(1 - np.tanh(z)**2)
        return value, j
    goal = np.zeros(192); goal[:30] = .1
    result = m.solve(goal, evaluate, linearize)
    assert calls == [17, 4]
    assert np.linalg.norm(result['logits']) <= m.RADIUS + 1e-12
    assert result['score'] < np.mean(goal**2)
    assert len(result['history']) == 4


def test_fail_closed_and_incumbent_ties():
    goal = np.zeros(192)
    r = m.solve(goal, lambda a: goal, lambda z: (goal, np.zeros((192, 30))))
    assert [x['chosen'] for x in r['history']] == [-1]*4
    with pytest.raises(ValueError):
        m.solve(goal, lambda a: goal, lambda z: (np.ones(192), np.zeros((192, 30))))
    with pytest.raises(ValueError):
        m.raw(np.ones(30)*100)
