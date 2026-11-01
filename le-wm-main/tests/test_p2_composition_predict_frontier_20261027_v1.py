"""NumPy/metadata/mock callbacks only; never neural or physical execution."""
from copy import deepcopy
import builtins
import numpy as np
import pytest
from experiments.spectral_reachability import p2_composition_predict_frontier_20261027_v1 as m


def base():
    right=np.eye(20);dls=np.linspace(-.1,.1,20)
    logits=np.stack([np.zeros(20)]+[s*dls for s in (.25,.5,1.)]+[s*r*right[k] for k in range(9) for r in m.RADII for s in (-1,1)])
    return dict(logits=logits.tolist(),commands=np.tanh(logits).astype(np.float32).reshape(40,2,5,2).tolist(),
        labels=['zero','dls_scale_.25','dls_scale_.5','dls_scale_1']+[f'mode_{k}_radius_{r}_sign_{s}' for k in range(9) for r in range(2) for s in (-1,1)],
        right_vectors=right.tolist(),dls=dict(params=dls.tolist()),goal_sha256='a'*64,projection_sha256='b'*64)


@pytest.mark.parametrize('root',m.ROOTS)
def test_nested80_and_original40_exact(root):
    bank=base();before=deepcopy(bank);a=m.candidate_bank(bank,root,40);b=m.candidate_bank(bank,root,80)
    assert bank==before and a['logits']==b['logits'][:40]==bank['logits']
    assert a['commands']==b['commands'][:40]==bank['commands'] and a['labels']==b['labels'][:40]
    assert b==m.candidate_bank(bank,root,80)
    assert np.max(np.linalg.norm(b['logits'],axis=1))<=m.CAP*(1+1e-12)
    assert np.array_equal(b['commands'],np.tanh(b['logits']).astype(np.float32).reshape(80,2,5,2))
    assert b['new_svd_calls']==b['new_predict_calls']==0 and not b['neural_ready']


def test_seed_only_new_tail_no_outcomes():
    a,b=(m.candidate_bank(base(),r) for r in m.ROOTS[:2])
    assert a['logits'][:40]==b['logits'][:40] and a['logits'][40:]!=b['logits'][40:]
    tail=np.asarray(a['logits'][40:]);assert np.array_equal(tail[:,9:],np.zeros((40,11)))
    assert np.allclose(tail[::2],-tail[1::2])
    assert len(a['tail_coefficients'])==10


def test_bound_bridge_rejects_foreign_bank_and_forged_Q(monkeypatch):
    b=base()
    assert not m.candidate_bank(b,m.ROOTS[0])['source_verified']
    with pytest.raises(ValueError):m.bound_candidate_bank(b,m.ROOTS[0])
    monkeypatch.setattr(m,'ROOT_BANK_DIGESTS',(m.digest(b),*m.ROOT_BANK_DIGESTS[1:]))
    assert m.bound_candidate_bank(b,m.ROOTS[0])['source_verified']
    with pytest.raises(ValueError):m.bound_candidate_bank(b,m.ROOTS[1])
    b['goal_sha256']='f'*64
    with pytest.raises(ValueError):m.bound_candidate_bank(b,m.ROOTS[0])


@pytest.mark.parametrize('damage',['shape','labels','mapping','sign','orthogonal','goal','dls','nan'])
def test_bank_drift_rejected(damage):
    b=base()
    if damage=='shape':b['logits'].pop()
    elif damage=='labels':b['labels'][0]='renamed'
    elif damage=='mapping':b['commands'][0][0][0][0]=.1
    elif damage=='sign':b['right_vectors'][0][0]=-1
    elif damage=='orthogonal':b['right_vectors'][1][0]=.1
    elif damage=='goal':b.pop('goal_sha256')
    elif damage=='dls':b['dls']['params'][0]=3
    else:b['logits'][0][0]=np.nan
    with pytest.raises(ValueError):m.candidate_bank(b,m.ROOTS[0])


@pytest.mark.parametrize('root,count',[('foreign',80),(m.ROOTS[0],True),(m.ROOTS[0],79),(m.ROOTS[0],120)])
def test_no_unfrozen_budget_or_root(root,count):
    with pytest.raises(ValueError):m.candidate_bank(base(),root,count)


def test_forward_entries_and_unique_not_sum_standalone():
    old=m.validate_entries(m.expected_entries(40),40);new=m.validate_entries(m.expected_entries(80),80)
    assert old['standalone_predict']==[88,168] and old['paired_unique_predict']==252
    assert new['standalone_predict']==[168,168] and new['paired_unique_predict']==332
    assert new['action_encoder_calls']==332 and new['context_tokens_per_entry_type']==996
    assert sum(new['standalone_predict'])==new['paired_unique_predict']+4
    assert new['standalone_Jacobian_calls']==[1,41] and new['standalone_mathematical_VJP_directions']==[192,7872]
    assert new['selection_evaluations']==120 and not new['total_resource_matched']
    assert m.summary()['four_root_new_point']['paired_unique_predict']==1328


@pytest.mark.parametrize('damage',['missing','extra','failed','tokens','batch','scope','nominal','order'])
def test_exact_calls_fail_closed(damage):
    es=m.expected_entries(80)
    if damage=='missing':es.pop()
    elif damage=='extra':es.append(deepcopy(es[-1]))
    elif damage=='failed':es[10]['completed']=False
    elif damage=='tokens':es[0]['context_tokens']=2
    elif damage=='batch':es[0]['samples']=2
    elif damage=='scope':es[0]['scope']=m.ARMS[0]
    elif damage=='nominal':es[0]['role']='hidden_AD_forward'
    else:es[0],es[1]=es[1],es[0]
    with pytest.raises(ValueError):m.validate_entries(es,80)


def legacy_inputs():
    from experiments.spectral_reachability import p2_composition_paired_20261027_v2 as old
    from experiments.spectral_reachability.d2p_composition_witness_v2 import PROVENANCE_KEYS
    calls=[]
    context=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
        latent_history=np.zeros((3,192)).tolist(),action_history=np.zeros((2,10)).tolist())
    def replay(c,actions):
        for packet in np.asarray(actions).reshape(-1,5,2):
            calls.append(packet.tolist());z=np.asarray(c['latent_history'][-1]).copy();z[:10]+=packet.reshape(10)
            c['latent_history']=c['latent_history'][1:]+[z.tolist()];c['action_history']=c['action_history'][1:]+[packet.reshape(10).tolist()]
        return c
    suffix_calls=[]
    def suffix(c,p,i):
        # Mock the TWO actual source terminal forwards; no AD engine invoked.
        for stage in ('nominal','ADforward'):
            suffix_calls.append((i,stage));replay(deepcopy(c),np.zeros((5,2)))
        return dict(suffix_logits=np.zeros(10).tolist(),input_context_sha256=old.digest(c))
    args=dict(bank=base(),root=dict(root_id='fixture',root_rgb_sha256='a'*64,goal_rgb_sha256='b'*64,context=context),
        provenance={k:'c'*64 for k in PROVENANCE_KEYS},
        action_domain=dict(name='raw',units='official PushT live command',low=[-1.,-1.],high=[1.,1.],action_block=5,normalization_sha256='d'*64),
        replay=replay,suffix=suffix,target_metric=lambda c:float(np.mean(np.asarray(c['latent_history'][-1])**2)),fixture=True)
    return old,args,calls,suffix_calls


@pytest.mark.parametrize('arm,forwards,suffix_forwards',[(m.ARMS[0],84,0),(m.ARMS[1],164,80)])
def test_ACTUAL_legacy_run_arm_with_counted_mock_callbacks(arm,forwards,suffix_forwards):
    old,args,calls,suffix_calls=legacy_inputs();r=old.run_arm(arm=arm,**args)
    assert r['complete'] and len(r['attempts'])==40
    assert len(calls)==forwards and len(suffix_calls)==suffix_forwards
    assert len(r['ledger'])==84 and sum(e['predict_samples'] for e in r['ledger'] if e['phase']=='planning')==80
    assert sum(e['predict_samples'] for e in r['ledger'] if e['phase']=='replay')==4
    # Whole ledger.call produces TWO packet-level adapter calls/events.
    assert [e['path'] for e in r['ledger'][-2:]]==[[42],[42]]
    assert [e['packet'] for e in r['ledger'][-2:]]==[0,1]
    assert sum(e['predict_samples'] for e in r['ledger'][-2:])==2


def test_old_runner_cannot_accept80_not_silent_reseal():
    old,args,calls,_=legacy_inputs();args['bank']=m.candidate_bank(base(),m.ROOTS[0],80)
    with pytest.raises(ValueError):old.run_arm(arm=m.ARMS[0],**args)
    assert calls==[]


def test_joint80_actual_mock_evaluations_not_padding():
    old,args,calls,_=legacy_inputs();bank=m.candidate_bank(base(),m.ROOTS[0],80);contexts=[];costs=[]
    for packets in bank['commands']:
        c=deepcopy(args['root']['context'])
        for packet in packets:c=args['replay'](c,packet)
        contexts.append(c);costs.append(args['target_metric'](c))
    winner=min(range(80),key=lambda k:costs[k]);raw=np.asarray(bank['commands'][winner]).reshape(10,2)
    original=deepcopy(args['root']['context']);seg=deepcopy(original)
    for packet in raw.reshape(2,5,2):seg=args['replay'](seg,packet)
    whole=args['replay'](deepcopy(original),raw)
    assert len(calls)==164 and whole==seg==contexts[winner]
    assert len(costs)==80 and min(costs)<=min(costs[:40])
    assert len(calls)==164  # no extra replay for diagnostic prefix40 minimum


def test_source_call_sites_and_drift(tmp_path):
    proof=m.legacy_source_contract();assert proof['verified_call_sites'] and not proof['old40_contract_accepts80']
    assert len(proof['remaining_runner_interfaces'])==6
    with pytest.raises(OSError):m.legacy_source_contract(tmp_path)


def test_bridge_pure_no_neural_or_physics_import(monkeypatch):
    original=builtins.__import__
    def guard(name,*a,**kw):
        assert name.split('.')[0] not in {'torch','gymnasium','pymunk','mujoco'},name
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',guard)
    assert m.summary()['new_neural_calls']==0
    assert m.candidate_bank(base(),m.ROOTS[0])['count']==80
