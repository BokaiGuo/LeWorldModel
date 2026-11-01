import numpy as np
import pytest
from experiments.spectral_reachability import p2_h15_gn_direction_guard_20261029_v1 as guard


def test_fixed_four_probes_and_wrong_jacobian_fails():
    z=np.linspace(-.2,.2,30);calls=[]
    j=np.zeros((192,30));j[:30]=np.diag(1-np.tanh(z)**2)
    result=dict(logits=z.tolist(),raw=guard.ad.gn.raw(z).tolist(),logit_jacobian=j.tolist())
    def evaluate(raw):
        calls.append(raw.copy());y=np.zeros(192);y[:30]=raw.reshape(30);return y
    checks=guard.verify(result,evaluate)
    assert len(calls)==4 and len(checks)==2
    assert [x['direction'] for x in checks]==[0,1]
    result['logit_jacobian']=(j*4).tolist()
    with pytest.raises(ValueError,match='directional difference'):
        guard.verify(result,evaluate)


def test_raw_mismatch_refuses_before_evaluation():
    z=np.zeros(30)
    result=dict(logits=z.tolist(),raw=np.ones((15,2)).tolist(),logit_jacobian=np.zeros((192,30)).tolist())
    with pytest.raises(ValueError,match='mapping'):
        guard.verify(result,lambda raw: pytest.fail('must not dispatch'))
