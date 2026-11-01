"""Non-neural numerical/metadata regressions; no CUDA or model allocation."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import p2_matched_sampling_20261027_v1 as m
from experiments.spectral_reachability.d2p_composition_witness_v2 import PROVENANCE_KEYS,digest


def transition(context,actions):
    for start in range(0,len(actions),5):
        packet=np.asarray(actions[start:start+5]).reshape(-1).tolist()
        v=np.asarray(context['latent_history'][-1])+packet[0]
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
    assert a['attempts'][0]['commands']==[[0.,0.],[0.,0.]]
    for e in a['attempts']:
        raw=np.asarray(e['actions']); assert raw.shape==(10,2)
        assert np.array_equal(raw[:5],np.repeat(raw[:1],5,axis=0))
        assert np.array_equal(raw[5:],np.repeat(raw[5:6],5,axis=0))
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


@pytest.mark.parametrize('commands', [np.zeros((1,2)),np.full((2,2),np.nan),np.full((2,2),2),np.zeros((2,2),bool)])
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
        compilation['rows'].append(dict(root_sha256=root['root_id'],compiled=dict(
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
            arms.append(r)
        report['rows'].append(dict(root_sha256=ready['root_sha256'],root=ready['root'],action_domain=ready['action_domain'],rows=arms))
    return report,protocol,compilation


def test_new_physical_consumer_accepts_all_eight_witnesses():
    args=consumer_fixture(); actions=m.selected_actions(*args)
    assert len(actions)==8 and all(x.shape==(10,2) for x in actions.values())


@pytest.mark.parametrize('corruption',['root','action','hash','metric','arm','missing_candidate','fixture','domain','provenance'])
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
    if corruption=='provenance':receipt['provenance']['metric_sha256']='0'*64
    with pytest.raises(ValueError):m.selected_actions(report,protocol,compilation)


def test_real_prepare_inputs_and_source_seal_without_cuda(tmp_path):
    comp=Path('results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json')
    ref=Path('results/pusht_p2_matched_composition4_v1_20261027_cuda/p2_matched_report.json')
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
    assert {m.TAG+'.py','test_'+m.TAG+'.py','P2_MATCHED_SAMPLING_20261027.md',
            'pusht_composition_cuda_replay_v2.py','d2p_composition_witness_v2.py',
            'run_pusht_program_search_cuda.py','d2p_adaptive_budget_v2.py'}<=mirrored
    with pytest.raises(FileExistsError):m.prepare(ref,comp,policy,out)
