"""Pure RAM/mock model responses and approved saved-data input projection."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_common_prefix_multiradius_cuda_20261027_v1 as m
import test_p1_constructive_common_prefix_probe_cuda_20261027_v1 as fixture


def transition(c,u):
    t=dict(action_mean=m.base.d2p.ACTION_MEAN,action_std=m.base.d2p.ACTION_STD)
    x=np.asarray(u,np.float32).reshape(10)
    z=np.zeros(192,np.float32); z[:10]=x+x*x
    return dict(c,latent_history=c['latent_history'][1:]+[z.astype(float).tolist()],
        action_history=[c['action_history'][-1],m.core.normalized_tokens(t,u)[0].tolist()])


def protocol():
    old=fixture.protocol(); slots=[]
    for pid,pos in enumerate(old['positions']):
        t=old['tasks'][pos['root_slot']]; c=deepcopy(t['root_context'])
        refs=[transition(c,m.physical.raw5(b['commands'])) for b in m.physical.probes()]
        slots.append(dict(position_id=pid,root_slot=pos['root_slot'],context=c,context_sha256=m.core.digest(c),references=refs,
            original_root=t['root_context']['latent_history'][-1],goal=t['goal']))
    return dict(slots=slots,bank=m.bank(),contract=m.contract())


def rows(p):
    return [dict(index=81*s['position_id']+b['index'],position_id=s['position_id'],probe=deepcopy(b),status='not_started') for s in p['slots'] for b in p['bank']]


def test_fixed81_bank_fourradii_ball_time_major_no_double_tanh():
    bank=m.bank(); assert len(bank)==81 and bank[0]['reference_index']==0
    assert np.array_equal(bank[0]['commands'],np.zeros((5,2)))
    for ri,rho in enumerate(m.RHOS):
        for j in range(10):
            for offset,sign in ((0,1),(1,-1)):
                b=bank[1+20*ri+2*j+offset]; x=np.asarray(b['logits'],np.float32)
                assert b['rho']==rho and b['sign']==sign and b['coordinate']==j
                assert np.count_nonzero(x)==1 and x[j]==sign*rho and np.linalg.norm(x)<=m.physical.LOGIT_CAP
                assert np.array_equal(b['commands'],np.tanh(x).reshape(5,2))
                assert b['reference_index']==(1+2*j+offset if rho==.25 else None)
    assert m.contract()['caps']==dict(predict=324,action_encoder=324)


def test324_genuine_predictions_reference84_samecontext_and_arithmetic():
    p=protocol(); out=rows(p); store=fixture.Store(); ledger=m.Ledger(store.event); calls=[]
    def predict(c,u):
        calls.append((deepcopy(c),u.copy())); ledger.call('action_encoder',lambda:None); ledger.call('predict',lambda:None)
        return transition(c,u)
    m.execute(p,predict,ledger,store,out)
    assert len(calls)==324 and all(r['status']=='predicted' for r in out)
    assert sum(r['reference_exact'] is True for r in out)==84
    assert ledger.counts()=={k:dict(attempted=324,completed=324) for k in ('predict','action_encoder')}
    for i,(c,u) in enumerate(calls): assert c==p['slots'][i//81]['context'] and np.array_equal(u,p['bank'][i%81]['commands'])
    a=m.arithmetic(p,out); assert a['planned']==324 and len(a['positions'])==4
    first=a['positions'][0]['radii'][0]
    assert np.asarray(first['B_192x10']).shape==(192,10) and first['B_relative_change_vs_smallest']==0.
    assert first['B_cosine_vs_smallest']==pytest.approx(1.)
    assert a['positions'][0]['radii'][3]['B_relative_change_vs_smallest']>0
    col=first['columns'][0]; u=np.float32(np.tanh(np.float32(.0625)))
    expected=float((np.float32(u+u*u)-np.float32(-u+u*u)))/(2*.0625)
    assert col['B'][0]==expected and col['curvature_per_rho2'][0]>0


def test_reference_mismatch_saved_not_tolerated_no_retry_partial_denominator():
    p=protocol(); out=rows(p); store=fixture.Store(); ledger=m.Ledger(store.event)
    def predict(c,u):
        ledger.call('action_encoder',lambda:None); ledger.call('predict',lambda:None)
        after=transition(c,u); after['latent_history'][-1][0]=float(np.nextafter(np.float32(0.),np.float32(1.)))
        return after
    with pytest.raises(ValueError,match='exact H1'): m.execute(p,predict,ledger,store,out)
    assert len(out)==324 and out[0]['status']=='reference_mismatch' and out[0]['reference_exact'] is False
    assert all(r['status']=='not_started' for r in out[1:]) and store.events[-1]['kind']=='H1_saved'
    a=m.arithmetic(p,out); assert a['positions'][0]['completed']==0
    assert a['positions'][0]['radii'][0]['B_192x10'] is None


def test_neural_failure_nonrefund_and_cap_reject_encoding_and_wrongorder():
    ledger=m.Ledger(lambda e:None); ledger.scope=(0,0)
    with pytest.raises(ValueError): ledger.call('encoder',lambda:None)
    with pytest.raises(ValueError): ledger.call('predict',lambda:None)
    with pytest.raises(RuntimeError): ledger.call('action_encoder',lambda:(_ for _ in ()).throw(RuntimeError('primary')))
    assert ledger.counts()['action_encoder']==dict(attempted=1,completed=0)
    with pytest.raises(ValueError): ledger.call('action_encoder',lambda:None)
    ledger=m.Ledger(lambda e:None); ledger.scope=(0,81)
    with pytest.raises(ValueError): ledger.call('action_encoder',lambda:None)


def test_missing_sides_and_zero_norm_keep_null_not_zero_fill():
    p=protocol(); out=rows(p)
    for i,r in enumerate(out):
        r.update(status='predicted',prediction_context=transition(p['slots'][i//81]['context'],m.physical.raw5(p['bank'][i%81]['commands'])))
    out[1]['status']='failed_partial'; p['slots'][0]['goal']=deepcopy(p['slots'][0]['original_root'])
    a=m.arithmetic(p,out); assert len(a['positions'][0]['radii'])==4
    assert a['positions'][0]['radii'][0]['B_192x10'] is None
    assert a['positions'][0]['radii'][0]['columns'][0]['B'] is None
    assert all(c['signed_unit'] is None for c in a['positions'][0]['radii'][1]['columns'])
    assert a['positions'][0]['radii'][1]['signed_l2'] is None


@pytest.fixture(scope='module')
def actual():
    path=m.ROOT/'results/p1_constructive_common_prefix_probe_cuda_20261027_v2_cuda_main01/report.json'
    r=m.read(path); b=m.read(path.parent/'binding.json'); p=m.read(b['prepared'])
    return path,r,p


def test_saved_model_projection_never_accesses_terminal_labels_analysis_or_physical_metrics(actual):
    _,r,p=actual; r=deepcopy(r)
    class InputsOnly(list):
        def __getitem__(self,key):
            if not isinstance(key,int) or not 0<=key<8: raise AssertionError('terminal labels forbidden')
            return super().__getitem__(key)
    r['images']=InputsOnly(r['images']); r['analysis']=object(); r['warm_backend_differences']=object()
    slots=m.model_slots(r,p)
    assert len(slots)==4 and all(len(s['references'])==21 for s in slots)
    for pid,s in enumerate(slots): assert s['context']==r['rows'][21*pid]['input_context']


def test_actual_material_RAM_only_external_SHA_and_size(actual):
    path,_,_=actual; p=m.material(str(path),m.PARENT_SHA)
    assert p['contract']['planned']==324 and len(p['slots'])==4 and len(p['bank'])==81
    assert len(m.canonical(p))<2*1024**2 and m.sha(m.parent.__file__)==m.SOURCE_SHA
    with pytest.raises(ValueError): m.material('/never/read','a'*64)


@pytest.mark.parametrize('damage',('context','command','identity','resources'))
def test_source_model_fields_tamper_failclosed(actual,damage):
    _,r,p=actual; r=deepcopy(r)
    if damage=='context': r['rows'][1]['input_context']['latent_history'][0][0]+=.01
    elif damage=='command': r['rows'][1]['commands'][0][0]=0.
    elif damage=='identity': r['rows'][0]['root_slot']=1
    else: r['resources']['predict']['completed']-=1
    with pytest.raises(ValueError): m.model_slots(r,p)
