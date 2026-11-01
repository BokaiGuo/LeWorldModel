"""Pure RAM F/Z/M regression tests; no prepare, neural library or simulator."""
from copy import deepcopy
from types import SimpleNamespace
import ast
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_live_overlay_fidelity_20261027_v2_cuda as m
from test_p1_constructive_live_overlay_fidelity_20261027_v1_cuda import Hook,MockProjection


def fixture():
    branches=[dict(branch=i*2+j,source_slot=i,arm=a,selection_digest=str(i*2+j),
        commands=np.linspace(-.75,.75,20,dtype=np.float32).reshape(10,2).astype(float).tolist()) for i in range(2) for j,a in enumerate(m.base.ARMS)]
    observed={(b,v,r):np.full(192,b*100+vi*10+j,np.float32) for b in range(4) for vi,v in enumerate(m.VIEWS) for j,r in enumerate(m.ROLES[:3])}
    labels={(b,v):np.full(192,b*100+vi*10+3,np.float32) for b in range(4) for vi,v in enumerate(m.VIEWS)}
    return branches,observed,labels


class Backend:
    """Array callback arithmetic, not a neural model or physical emulator."""
    def __init__(self):self.calls=[]
    def predict(self,c,a,control,sink):
        self.calls.append((deepcopy(c),a.copy(),control))
        tokens=m.action_tokens(c,a);embedding=np.resize(tokens,(1,3,192)).astype(np.float32)
        conditioned=m.mask_embedding(embedding,control)
        out=np.asarray(c['latent_history'],np.float32)+np.float32(np.mean(conditioned)*.01)
        sink(dict(kind='mock_control',control=control))
        return dict(predicted3=out.astype(float).tolist(),action_tokens=tokens.astype(float).tolist(),
            embedding_before=embedding[0].astype(float).tolist(),embedding_used=conditioned[0].astype(float).tolist())


def test_fixed24_order_and_own_observed_context():
    branches,observed,labels=fixture();requests=m.make_requests(branches,observed)
    assert len(requests)==24
    assert [(r['branch'],r['view'],r['control']) for r in requests]==[(b,v,c) for b in range(4) for v in m.VIEWS for c in m.CONTROLS]
    for r in requests:
        assert np.array_equal(r['context']['latent_history'],[observed[(r['branch'],r['view'],role)] for role in m.ROLES[:3]])
        assert set(r)==set('index branch source_slot arm view control context commands raw_between'.split())


def test_zero_is_raw_zero_not_normalized_zero_and_history_unchanged():
    branches,observed,_=fixture();f,z,masked=m.make_requests(branches,observed)[:3]
    assert f['context']==z['context']==masked['context'] and f['commands']==masked['commands']
    assert not np.asarray(z['commands']).any()
    ft,zt,mt=[m.action_tokens(r['context'],r['commands']) for r in (f,z,masked)]
    assert np.array_equal(ft[:2],zt[:2]) and np.array_equal(ft,mt)
    assert np.any(zt[-1]!=0) and np.array_equal(zt[-1],m.base.d2p.normalized_history(np.zeros((5,2),np.float32))[0])
    assert not np.array_equal(ft[-1],zt[-1])


def test_history_uses_actual_zero_then_actual_prefix_time_major():
    branches,observed,_=fixture();r=m.make_requests(branches,observed)[0]
    raw=np.asarray(r['raw_between'],np.float32)
    assert not raw[:5].any() and np.array_equal(raw[5:],np.asarray(branches[0]['commands'],np.float32)[:5])
    expected=(raw.reshape(2,10)-np.tile(m.base.d2p.ACTION_MEAN,5))/np.tile(m.base.d2p.ACTION_STD,5)
    assert np.array_equal(expected,r['context']['action_history'])


@pytest.mark.parametrize('control',['F','Z','M'])
def test_mask_only_last_embedding_no_inplace(control):
    e=np.arange(576,dtype=np.float32).reshape(1,3,192)+1;original=e.copy();out=m.mask_embedding(e,control)
    assert np.array_equal(e,original) and not np.shares_memory(e,out)
    assert np.array_equal(out[:,:2],e[:,:2])
    assert np.array_equal(out[:,-1],np.zeros((1,192),np.float32) if control=='M' else e[:,-1])


@pytest.mark.parametrize('shape',[(3,192),(1,2,192),(1,3,10)])
def test_wrong_embedding_shape_rejected(shape):
    with pytest.raises(ValueError):m.mask_embedding(np.zeros(shape,np.float32),'M')


def test_all24_fresh_calls_even_identical_inputs():
    branches,observed,labels=fixture()
    for k in observed:observed[k].fill(1)
    requests=m.make_requests(branches,observed);backend=Backend();events=[]
    outputs=m.predict_all(requests,backend,events.append)
    assert len(backend.calls)==len(outputs)==24
    assert sum(e['kind']=='H1_attempt' for e in events)==sum(e['kind']=='H1_completed' for e in events)==24
    assert all(backend.calls[i][0]==backend.calls[i+1][0] for i in range(0,24,3))
    assert all(o['request_index']==i for i,o in enumerate(outputs))


def test_terminal_label_firewall_and_evaluation_no_neural():
    branches,observed,labels=fixture();requests=m.make_requests(branches,observed);backend=Backend()
    outputs=m.predict_all(requests,backend,lambda e:None);before=deepcopy(requests)
    first=m.evaluate(requests,outputs,labels)
    labels[(0,'off')].fill(10000)
    second=m.evaluate(requests,outputs,labels)
    assert requests==before and len(backend.calls)==24
    assert first[0]['errors']!=second[0]['errors']
    assert first[1:]==second[1:]  # no cross-view or cross-branch target contamination


def test_labels_cannot_be_added_to_observed_whitelist():
    branches,observed,labels=fixture();observed[(0,'off','terminal10')]=labels[(0,'off')]
    with pytest.raises(ValueError,match='observed-only'):m.make_requests(branches,observed)


@pytest.mark.parametrize('fault',['reorder','missing','duplicate','bad_action','bad_latent'])
def test_bad_fixed_source_inputs_fail(fault):
    branches,observed,_=fixture()
    if fault=='reorder':branches.reverse()
    elif fault=='missing':observed.pop(next(iter(observed)))
    elif fault=='duplicate':branches[1]=deepcopy(branches[0])
    elif fault=='bad_action':branches[0]['commands'][0][0]=1.1
    else:observed[(0,'off','warmup5')][0]=np.nan
    with pytest.raises(ValueError):m.make_requests(branches,observed)


def test_arithmetic_metrics_and_signed_fzm_differences():
    branches,observed,labels=fixture();requests=m.make_requests(branches,observed);outputs=[]
    for r in requests:
        current=np.asarray(r['context']['latent_history'][-1],np.float32)
        delta={'F':.25,'Z':0.,'M':-.5}[r['control']]
        predicted=np.tile(current+np.float32(delta),(3,1))
        outputs.append(dict(request_index=r['index'],predicted3=predicted.astype(float).tolist()))
    groups=m.evaluate(requests,outputs,labels)
    assert len(groups)==8
    for g in groups:
        assert g['errors']==dict(F=.5625,Z=1.,M=2.25)
        assert g['persistence_mse']==1 and g['Z_minus_F']==.4375 and g['M_minus_F']==1.6875
        assert g['F_vs_Z_output_mse']==.0625 and g['F_vs_M_output_mse']==.5625
        assert g['P1_admission'] is False and 'no Z/M physical truth' in g['labels']


def test_failure_stops_no_retry_and_durable_attempt():
    branches,observed,_=fixture();requests=m.make_requests(branches,observed);events=[]
    class Bad(Backend):
        def predict(self,*a):
            if len(self.calls)==2:raise ValueError('sentinel')
            return super().predict(*a)
    backend=Bad()
    with pytest.raises(ValueError,match='sentinel'):m.predict_all(requests,backend,events.append)
    assert len(backend.calls)==2
    assert sum(e['kind']=='H1_attempt' for e in events)==3 and sum(e['kind']=='H1_completed' for e in events)==2


def test_changed_condition_order_rejected_before_calls():
    branches,observed,_=fixture();requests=m.make_requests(branches,observed);requests[1],requests[2]=requests[2],requests[1];b=Backend()
    with pytest.raises(ValueError):m.predict_all(requests,b,lambda e:None)
    assert not b.calls


@pytest.mark.parametrize('resource',['predict','action_encoder'])
def test_actual_ledger24_not_old8_or20(resource):
    ledger=m.PredictLedger(lambda e:None,lambda:None)
    for _ in range(24):ledger.end(ledger.begin(resource))
    with pytest.raises(ValueError,match='fixed24'):ledger.begin(resource)
    assert ledger.counts()[resource]==dict(attempted=24,completed=24)


@pytest.mark.parametrize('shape,valid',[((3,192),True),((1,3,192),False)])
def test_predict_and_flattened_projection24_receipts(shape,valid):
    projection=MockProjection();model=SimpleNamespace(pred_proj=projection,predict=lambda x:projection(x));original=model.predict
    x=SimpleNamespace(shape=shape,device=SimpleNamespace(type='cuda'));events=[]
    if valid:
        with m.prediction_receipts(model,events.append,lambda:None) as receipts:
            for _ in range(24):model.predict(x)
            with pytest.raises(ValueError,match='24 cap'):model.predict(x)
        assert len(receipts)==48 and all(e['completed'] for e in receipts)
        assert sum(e.get('tokens',0) for e in receipts)==72
    else:
        with pytest.raises(ValueError,match='flattened'):
            with m.prediction_receipts(model,events.append,lambda:None):model.predict(x)
    assert model.predict is original and not projection.pre and not projection.post


def test_build_only_reuses_saved_v1_input_not_v1_run(monkeypatch):
    original=dict(schema='old',contract=dict(predict=8),arguments=dict(paired_report='PAIR',paired_sha='HASH'),
        bindings={},images=[dict(index=i) for i in range(32)])
    seen=[]
    monkeypatch.setattr(m.previous,'build',lambda *a:(seen.append(a),deepcopy(original))[1])
    def hashfile(path):
        if str(path)==m.previous.__file__:return m.OLD_SHA
        if str(path)==m.render.__file__:return m.RENDER_SHA
        return 'mock-source-sha'
    monkeypatch.setattr(m,'sha',hashfile)
    p=m.build('PAIR','HASH')
    assert seen==[('PAIR','HASH')] and p['schema']==m.NAME+'.prepare' and p['contract']['predict']==24
    assert original['contract']['predict']==8 and len(p['bindings'])==3


def test_budget_and_no_unrequested_scientific_gate():
    c=m.contract();assert c['encoder']==c['projector']==32
    assert c['predict']==c['action_encoder']==c['model_predict_entries']==c['pred_proj_calls']==24
    assert c['action_encoder_packet_tokens']==c['pred_proj_token_samples']==72
    assert c['AD']==c['physics']==0 and c['P1_admission'] is c['P2_admission'] is c['selection'] is False
    assert c['scientific_pass_threshold'] is None


def test_native_backend_cuda_only_without_importing_torch():
    with pytest.raises(ValueError,match='CUDA'):m.NativeBackend(None,SimpleNamespace(type='cpu'))


def test_no_simulator_or_legacy_run_calls_in_new_source():
    tree=ast.parse(m.Path(m.__file__).read_text())
    imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert not any('gymnasium' in name or 'pymunk' in name for name in imports)
    native=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='NativeBackend')
    attrs=[n.func.attr for n in ast.walk(native) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert attrs.count('action_encoder')==1 and attrs.count('predict')==1 and 'predictor' not in attrs
