import numpy as np
import pytest
from experiments.spectral_reachability import p2_h15_gradient_solver_20261029_v1 as m
from experiments.spectral_reachability import p2_h15_projected_gn_20261029_v1 as gn


def test_gradient_is_fixed_distinct_from_gn_and_bounded():
    calls=[0,0];goal=np.zeros(192);goal[:30]=.1
    def evaluate(a):
        calls[0]+=1;out=np.zeros(192);out[:30]=np.asarray(a).reshape(30);return out
    def linearize(z):
        calls[1]+=1;out=np.zeros(192);out[:30]=m.raw(z).reshape(30)
        j=np.zeros((192,30));j[:30]=np.diag(1-np.tanh(z)**2);return out,j
    result=m.solve(goal,evaluate,linearize)
    assert calls==[17,4] and len(result['history'])==4
    assert np.linalg.norm(result['logits'])<=m.RADIUS+1e-12
    assert result['score']<np.mean(goal**2)
    other=gn.solve(goal,lambda a:np.pad(np.asarray(a).reshape(30),(0,162)),
        lambda z:(np.pad(gn.raw(z).reshape(30),(0,162)),np.vstack([np.diag(1-np.tanh(z)**2),np.zeros((162,30))])))
    assert not np.array_equal(result['logits'],other['logits'])


def test_invalid_incumbent_fails_closed():
    with pytest.raises(ValueError):
        m.solve(np.zeros(192),lambda a:np.zeros(192),lambda z:(np.ones(192),np.zeros((192,30))))
