"""RAM core contracts only; artificial arrays are not control evidence."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import capability_suffix_factorial_cuda_MAIN_20261027_v2 as m


def fixture(scale=1.):
    goal=np.zeros(192);goal[:10]=.02
    j=np.zeros((192,10));j[:10]=np.eye(10)*scale
    case=dict(scale=scale,goal=goal.tolist())
    calls=[];ops=[]
    def terminal(u):
        calls.append(np.asarray(u).copy())
        result=np.zeros(192,np.float32);result[:10]=scale*np.tanh(np.asarray(u,np.float32))
        return dict(prediction=result,raw=(scale*np.tanh(np.asarray(u,np.float32))).reshape(5,2),raw_receipt=dict(test=len(calls)))
    def operation(kind,fn):ops.append(kind);return fn()
    return case,terminal,lambda:j,calls,ops,operation


@pytest.mark.parametrize('scale',[1.,.1])
def test_ten_candidates_exact_replay_and_own_scale(scale):
    c,t,j,calls,ops,operation=fixture(scale)
    before=deepcopy(c);r=m.compile_case(c,t,j,operation)
    assert c==before and len(r['candidates'])==10 and r['replay_exact']
    assert len(calls)==12 and ops==['jacobian','svd','svd']
    for row in r['candidates']:
        np.testing.assert_allclose(row['raw'],scale*np.tanh(np.asarray(row['params'],np.float32)).reshape(5,2),rtol=0,atol=0)
        assert np.max(np.abs(row['raw']))<=scale
        assert row['Q']==m.cost(row['predicted'],c['goal'])
    assert r['winner']==min(r['candidates'],key=lambda x:(x['Q'],x['index']))


def test_tie_uses_original_bank_order_and_preserves_duplicates():
    c,t,j,calls,ops,operation=fixture();c['goal']=[0.]*192
    def flat(u):return dict(prediction=np.zeros(192,np.float32),raw=np.tanh(np.asarray(u,np.float32)).reshape(5,2),raw_receipt={})
    r=m.compile_case(c,flat,lambda:np.zeros((192,10)),operation)
    assert len(r['candidates'])==10 and r['winner']['index']==0


def test_independent_replay_change_rejected():
    c,t,j,calls,ops,operation=fixture()
    def drift(u):
        value=t(u)
        if len(calls)==12:value['prediction'][0]+=1
        return value
    with pytest.raises(ValueError):m.compile_case(c,drift,j,operation)


@pytest.mark.parametrize('bad',[np.zeros((191,10)),np.full((192,10),np.nan)])
def test_jacobian_shape_and_nonfinite_refused(bad):
    c,t,j,calls,ops,operation=fixture()
    with pytest.raises(ValueError):m.compile_case(c,t,lambda:bad,operation)


def test_contract_counts_and_no_scientific_admission():
    c=m.contract()
    assert c['predict_API']==24*13 and c['jacobians']==24 and c['SVD']==48
    assert c['mathematical_VJP_directions']==24*192 and not c['P1_admission']
    assert not c['total_FLOPs_matched'] and c['fresh_roots']==0


def test_verify_external_seal_and_dependency_change(tmp_path):
    f=tmp_path/'dependency';f.write_text('source')
    p=dict(schema=m.NAME+'.prepare',contract=m.contract(),cases=[dict(index=i) for i in range(24)],bindings={str(f):m.sha(f)})
    prepared=tmp_path/'prepare.json';prepared.write_bytes(m.source.rgb.canonical(p));h=m.sha(prepared)
    assert len(m.verify(prepared,h)['cases'])==24
    with pytest.raises(ValueError):m.verify(prepared,'0'*64)
    f.write_text('drift')
    with pytest.raises(ValueError):m.verify(prepared,h)

def test_actual_backend_raw_is_preserved_not_numpy_regenerated():
    c,t,j,calls,ops,operation=fixture()
    def backend(u):
        value=t(u)
        value['raw']=np.nextafter(value['raw'],np.float32(0.5))
        return value
    r=m.compile_case(c,backend,j,operation)
    for row in r['candidates']:
        expected=np.nextafter(np.tanh(np.asarray(row['params'],np.float32)).reshape(5,2),np.float32(.5))
        assert np.array_equal(np.asarray(row['raw'],np.float32),expected)
        assert row['raw_receipt']['test']>=2


def test_independent_raw_replay_drift_rejected():
    c,t,j,calls,ops,operation=fixture()
    def backend(u):
        value=t(u)
        if len(calls)==12:
            value['raw']=np.nextafter(value['raw'],np.float32(.5))
        return value
    with pytest.raises(ValueError,match='raw replay exact'):
        m.compile_case(c,backend,j,operation)


def test_terminal_array_without_actual_raw_receipt_rejected():
    c,t,j,calls,ops,operation=fixture()
    with pytest.raises(ValueError,match='actual terminal record'):
        m.compile_case(c,lambda u:np.zeros(192,np.float32),j,operation)
