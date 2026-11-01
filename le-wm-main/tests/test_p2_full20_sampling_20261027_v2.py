"""Non-neural numerical/metadata regressions; no CUDA or model allocation."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import p2_full20_sampling_20261027_v2 as m
from experiments.spectral_reachability.d2p_composition_witness_v2 import PROVENANCE_KEYS,digest


def transition(context,actions):
    for start in range(0,len(actions),5):
        packet=np.asarray(actions[start:start+5]).reshape(-1).tolist()
        v=np.asarray(context['latent_history'][-1])+sum((i+1)*v for i,v in enumerate(packet))/55
        context['latent_history']=context['latent_history'][1:]+[v.tolist()]
        context['action_history']=context['action_history'][1:]+[packet]
    return context


def inputs(arm='cem',dim=1):
    c=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=dim,
        action_dim=10,latent_history=[[0.]*dim for _ in range(3)],action_history=[[0.]*10 for _ in range(2)])
    return dict(arm=arm,seed=42,root=dict(root_id='a'*64,root_rgb_sha256='b'*64,goal_rgb_sha256='c'*64,context=c),
        provenance={k:'d'*64 for k in PROVENANCE_KEYS},
        action_domain=dict(name='raw',units='official PushT live command',low=[-1.,-1.],high=[1.,1.],
                           action_block=5,normalization_sha256='e'*64),
        target_metric=lambda c:float(np.mean((np.asarray(c['latent_history'][-1])-.5)**2)),replay=transition)


@pytest.mark.parametrize('arm',m.ARMS)
def test_exact40_full_h2_80_plus4_and_determinism(arm):
    args=inputs(arm); original=deepcopy(args['root']); seen=[]
    a=m.run_fixture_arm(**args,journal=seen.append); b=m.run_fixture_arm(**args)
    assert args['root']==original
    assert a['prediction_budget_matched'] and a['fixture_only']
    assert not a['P1_admission'] and not a['P2_admission'] and not a['fair_total_budget_comparison']
    assert len(seen)==len(a['attempts'])==40
    assert a['planning_predict_samples']==80 and a['replay_predict_samples']==4
    assert len(a['ledger'])==84 and all(e['completed'] for e in a['ledger'])
    assert a['attempts']==b['attempts'] and a['selected_candidate_id']==b['selected_candidate_id']
    assert a['attempts'][0]['commands']==np.zeros((2,5,2)).tolist()
    for e in a['attempts']:
        raw=np.asarray(e['actions']); assert raw.shape==(10,2)
        assert np.asarray(e['commands']).shape==(2,5,2)
        assert np.array_equal(raw,np.asarray(e['commands']).reshape(10,2))
        assert e['proposal_dimensions']==20
    proposals=np.asarray([e['actions'] for e in a['attempts'][1:11]]).reshape(10,20)
    assert np.all(proposals.std(axis=0)>0)
    assert not np.array_equal(proposals[0].reshape(10,2)[:5],np.repeat(proposals[0,None,:2],5,axis=0))
    w=a['witness']; assert w['independent_full_replay_exact']
    assert transition(deepcopy(args['root']['context']),w['receipt']['actions'])==w['receipt']['terminal_context']


def test_constant_metric_stable_first_tie_and_no_extra_mean():
    args=inputs(); args['target_metric']=lambda c:1.
    r=m.run_fixture_arm(**args)
    assert r['selected_candidate_id']=='cem_00'
    assert len(r['attempts'])==40


@pytest.mark.parametrize('which',['predict','metric','verification'])
def test_failures_retained_no_padding_or_refund(which):
    args=inputs(); count=0
    def replay(context,actions):
        nonlocal count
        count+=1
        if count==(1 if which=='predict' else 81) and which!='metric':
            raise RuntimeError('fixture failure')
        return transition(context,actions)
    args['replay']=replay
    if which=='metric':args['target_metric']=lambda c:float('nan')
    r=m.run_fixture_arm(**args)
    assert len(r['attempts'])==40 and not r['prediction_budget_matched']
    if which=='metric':
        assert r['failed_candidates']==40 and r['planning_predict_samples']==80 and r['witness'] is None
    elif which=='predict':
        assert r['failed_candidates']==1 and not r['ledger'][0]['completed']
        assert r['planning_predict_samples']==79
    else:
        assert r['replay_error'] and r['selected_candidate_id'] is not None
        assert r['replay_predict_samples']==1


@pytest.mark.parametrize('commands', [np.zeros((1,2)),np.full((2,5,2),np.nan),np.full((2,5,2),2),np.zeros((2,5,2),bool),np.zeros((2,2)),np.zeros((10,2))])
def test_bad_raw(commands):
    with pytest.raises(ValueError):m.raw_actions(commands)


def test_metadata_bad_before_replay_and_no_cpu_adapter():
    args=inputs();args['provenance']={}
    args['replay']=lambda c,a:pytest.fail('invalid metadata dispatched')
    with pytest.raises(ValueError):m.run_fixture_arm(**args)
    from types import SimpleNamespace
    with pytest.raises(ValueError,match='CUDA'):
        m.RootAnchoredCudaReplay(object(),SimpleNamespace(type='cpu'))


def consumer_fixture():
    protocol=dict(schema=m.TAG+'.protocol',schedule=m.SCHEDULE,prepared=[])
    compilation=dict(rows=[])
    for i in range(4):
        args=inputs(dim=192);args['root']['root_id']=str(i)*64
        root=args['root']; domain=args['action_domain']
        compilation['rows'].append(dict(root_index=i,root_sha256=root['root_id'],compiled=dict(
            input_context=dict(emb=[root['context']['latent_history']],actions=[[[0.]*10]+root['context']['action_history']]),
            goal_vectors=[[.5]*192,[.5]*192],scoring={'metric':{'projection':np.eye(192).tolist()}})))
        protocol['prepared'].append(dict(root_sha256=root['root_id'],root=root,action_domain=domain,
                                        metric_sha256='f'*64,seed=42))
    report=dict(schema=m.TAG+'.report',status='calibration_completed',gates={'fixture':True},
                protocol_sha256=digest(protocol),rows=[],source_seals=[{'manifest_sha256':'1'*64}])
    for ready in protocol['prepared']:
        arms=[]
        for arm in m.ARMS:
            args=inputs(arm,dim=192);args['root']=ready['root']
            args['provenance']=m.roots_provenance(source_sha='1'*64,protocol=protocol,ready=ready)
            r=m.run_fixture_arm(**args)
            r['fixture_only']=False  # METADATA TEST ONLY; never written as real evidence
            r['actual_model_entries']=[dict(kind=kind,samples=1,context_tokens=3,completed=True,
                scope=dict(arm=arm,phase=e['phase'],path=e['path']))
                for e in r['ledger'] for kind in ('action_encoder','predict')]
            arms.append(r)
        report['rows'].append(dict(root_sha256=ready['root_sha256'],root=ready['root'],action_domain=ready['action_domain'],rows=arms))
    return report,protocol,compilation


def test_new_physical_consumer_accepts_all_eight_witnesses():
    args=consumer_fixture(); actions=m.selected_actions(*args)
    assert len(actions)==8 and all(x.shape==(10,2) for x in actions.values())


@pytest.mark.parametrize('corruption',['root','action','hash','metric','arm','missing_candidate','fixture','domain','provenance','actual','candidate_shape'])
def test_physical_consumer_rejects_tampering(corruption):
    report,protocol,compilation=consumer_fixture()
    row=report['rows'][0];arm=row['rows'][0];receipt=arm['witness']['receipt']
    if corruption=='root':row['root']=dict(row['root'],root_id='bad')
    if corruption=='action':receipt['actions'][0][0]+=.1
    if corruption=='hash':receipt['receipt_sha256']='0'*64
    if corruption=='metric':arm['selected_model_cost']+=1
    if corruption=='arm':row['rows'][1]['arm']='cem'
    if corruption=='missing_candidate':arm['attempts'].pop()
    if corruption=='fixture':arm['fixture_only']=True
    if corruption=='domain':row['action_domain']=dict(row['action_domain'],action_block=1)
    if corruption=='actual':arm['actual_model_entries'].pop()
    if corruption=='candidate_shape':arm['attempts'][2]['commands']=[[0.,0.],[0.,0.]]
    if corruption=='provenance':receipt['provenance']['metric_sha256']='0'*64
    with pytest.raises(ValueError):m.selected_actions(report,protocol,compilation)


def test_real_prepare_inputs_and_source_seal_without_cuda(tmp_path,monkeypatch):
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    monkeypatch.setattr(base,'require_cuda_allocation',lambda *a,**k:pytest.fail('CUDA requested'))
    monkeypatch.setattr(base,'load_compatible_pretrained',lambda *a,**k:pytest.fail('model requested'))
    comp=Path('results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json')
    ref=Path('results/pusht_p2_matched_composition4_v1_20261027_cuda/p2_matched_report.json')
    assert m.DEFAULT_PAIRED_REPORT==ref.resolve() and m.DEFAULT_PAIRED_REPORT.is_file()
    policy=Path('/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm')
    out=tmp_path/m.TAG
    m.prepare(ref,comp,policy,out)
    _,p,prepared,source=m.load_prepared(out)
    assert len(p['prepared'])==len(source['rows'])==4
    assert p['schedule']['candidates']==40
    assert not (out/'started.json').exists()
    assert {r['root_sha256'] for r in p['prepared']}=={r['root_sha256'] for r in source['rows']}
    mirrored=set()
    for seal in prepared['source_seals']:
        manifest=json.loads((Path(seal['output_dir'])/'manifest.json').read_text())
        mirrored.update(Path(f['source']).name for f in manifest['files'])
    assert {m.TAG+'.py','test_'+m.TAG+'.py',m.TAG+'.md','pusht_full_packet_compiler_20261027.py',
            'pusht_composition_cuda_replay_v2.py','d2p_composition_witness_v2.py',
            'run_pusht_program_search_cuda.py','d2p_adaptive_budget_v2.py'}<=mirrored
    with pytest.raises(FileExistsError):m.prepare(ref,comp,policy,out)


def test_time_major_full20_mapping_and_bounded_input():
    commands=np.arange(20,dtype=float).reshape(2,5,2)/20
    assert m.raw_actions(commands)==commands.reshape(10,2).astype(np.float32).tolist()
    for i in range(20):
        delta=np.zeros((2,5,2));delta.reshape(-1)[i]=1
        assert np.flatnonzero(np.asarray(m.raw_actions(delta)).reshape(-1)).tolist()==[i]
    assert m.raw_actions(np.full((2,5,2),-1))==[[-1.,-1.]]*10


def test_normalization_non_neural_tensor_arithmetic():
    # CPU tensor arithmetic only: no model instance, training or neural forward.
    import torch
    from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
    raw=np.arange(20,dtype=np.float32).reshape(2,5,2)/20
    actual=p0.packets.normalize_live_packet(torch.tensor(raw),
        mean=p0.PUBLIC_ACTION_MEAN,std=p0.PUBLIC_ACTION_STD).numpy()
    mean=np.asarray(p0.PUBLIC_ACTION_MEAN);std=np.asarray(p0.PUBLIC_ACTION_STD)
    if mean.shape==(2,):mean=np.tile(mean,5)
    if std.shape==(2,):std=np.tile(std,5)
    np.testing.assert_array_equal(actual,(raw.reshape(2,10)-mean)/std)
    assert actual.shape==(2,10)


def test_export_all_eight_durable_plans():
    args=consumer_fixture();plans=m.plans_from_saved(*args)
    assert len(plans)==8 and len({(p['root_sha256'],p['method']) for p in plans})==8
    assert [p['method'] for p in plans[:4]]==['cem','random','random','cem']
    for plan in plans:
        assert plan['planned_commands']==10 and plan['planned_horizon_tokens']==2
        assert m.digest({k:v for k,v in plan.items() if k!='selection_sha256'})==plan['selection_sha256']
        assert np.asarray(plan['commands']).shape==(10,2)


def test_actual_entry_wrapper_records_failures_and_restores():
    from types import SimpleNamespace
    class Encoder:
        def forward(self,x):
            return x
    class Fake:
        def __init__(self):self.action_encoder=Encoder()
        def predict(self,emb,act):
            raise RuntimeError('metadata-only failure')
    model=Fake(); replay=SimpleNamespace(_p2_scope={'arm':'cem','phase':'planning','path':[0]})
    metadata=SimpleNamespace(shape=(1,3,10))
    with m.actual_entries(model,replay) as events:
        model.action_encoder.forward(metadata)
        with pytest.raises(RuntimeError):model.predict(metadata,metadata)
    assert 'predict' not in vars(model) and 'forward' not in vars(model.action_encoder)
    assert [e['kind'] for e in events]==['action_encoder','predict']
    assert events[0]['completed'] and not events[1]['completed']
    assert events[1]['error_type']=='RuntimeError'
    assert not m.entries_match(events,[{'phase':'planning','path':[0]}])


def test_actual_entries_and_budget_together_metadata_only():
    from types import SimpleNamespace
    from experiments.spectral_reachability.d2p_adaptive_budget_v2 import PredictionBudget,Binding
    class Encoder:
        def forward(self,x):return x
    class Fake:
        def __init__(self):self.action_encoder=Encoder()
        def predict(self,emb,act):return emb
    model=Fake();replay=SimpleNamespace(_p2_scope=None)
    metadata=SimpleNamespace(shape=(1,3,10),device=SimpleNamespace(type='cuda',index=0))
    ledger=[]
    with m.actual_entries(model,replay) as entries, PredictionBudget(model,
            binding=Binding('a'*64,m.TAG,'b'*64,'c'*64),method='cem',sample_limit=84) as budget:
        for name,count in [('planning',80),('evaluation',4)]:
            with budget.phase(name,method='cem',horizon_tokens=2):
                for i in range(count):
                    phase='planning' if name=='planning' else 'replay'
                    replay._p2_scope=dict(arm='cem',phase=phase,path=[i])
                    model.predict(metadata,model.action_encoder.forward(metadata))
                    ledger.append(dict(phase=phase,path=[i],completed=True,predict_calls=1,predict_samples=1))
        receipt=budget.receipt()
    assert m.entries_match(entries,ledger,'cem')
    assert not m.entries_match(entries,ledger,'random')
    assert len(entries)==168 and receipt['attempted_predict_samples']==84
    assert [(p['predict_calls'],p['predict_samples']) for p in receipt['phases']]==[(80,80),(4,4)]
    assert 'predict' not in vars(model) and 'forward' not in vars(model.action_encoder)


def test_second_packet_failure_is_not_refunded():
    args=inputs(); count=0
    def failing(context,actions):
        nonlocal count
        count+=1
        if count==2:raise RuntimeError('second packet failed')
        return transition(context,actions)
    args['replay']=failing
    result=m.run_fixture_arm(**args)
    assert not result['prediction_budget_matched'] and result['failed_candidates']==1
    assert result['planning_predict_samples']==80
    assert result['ledger'][0]['completed'] and not result['ledger'][1]['completed']


def test_independent_replay_mismatch_retained():
    args=inputs();count=0
    def mismatched(context,actions):
        nonlocal count
        count+=1
        result=transition(context,actions)
        if count==84:result['latent_history'][-1][0]+=.125
        return result
    args['replay']=mismatched
    result=m.run_fixture_arm(**args)
    assert result['planning_predict_samples']==80 and result['replay_predict_samples']==4
    assert result['witness'] is None and result['replay_error']['error_type']=='ValueError'
    assert not result['prediction_budget_matched']

