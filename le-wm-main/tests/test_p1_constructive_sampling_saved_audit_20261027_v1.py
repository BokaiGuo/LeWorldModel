"""Independent saved-array arithmetic and in-memory corruptions, no neural/tmp."""
import ast
from copy import deepcopy
from pathlib import Path
import pytest
from experiments.spectral_reachability import p1_constructive_sampling_saved_audit_20261027_v1 as m


@pytest.fixture(scope='module')
def saved():
    report=m.read(m.REPORT,m.REPORT_SHA);p=m.read(report['prepared'],m.PREPARE_SHA);directory=m.REPORT.parent
    wrapped=[m.io.parse(line) for line in m.pure.load(directory/'events.jsonl.gz').splitlines()]
    branches=[]
    for row in report['rows']:
        s=row['source_slot'];a=row['arm'];slot=p['slots'][s]
        t=m.pure.export_task(slot['task_export'],slot['task_sha256'])
        group=[e['event'] for e in wrapped if e['source_slot']==s and e['arm']==a]
        branches.append((t,slot,a,m.read(directory/row['result_file']),row,m.read(directory/row['plan_file']),group))
    return report,p,branches


def test_actual_complete_independent_audit():
    r=m.audit()
    assert r['totals']==dict(branches=4,candidates=256,fullH2_evaluations=268,predict=536,action_encoder=536)
    assert all(r['checks'].values()) and r['neural_calls']==r['physics_calls']==0
    assert not r['P1_admission'] and not r['producer_optimize_or_consume_imported']


@pytest.mark.parametrize('index,winner',[(0,2),(1,53),(2,33),(3,52)])
def test_each_actual_branch(saved,index,winner):
    r=m.audit_branch(*saved[2][index]);assert r['winner_index']==winner and r['predict']==134


@pytest.mark.parametrize('bad',[
    'seed','source_slot','arm','taskid','cost','terminal','candidate_logits','candidate_commands',
    'candidate_index','candidate_iteration','missing_candidate','update_mean','update_std','elite',
    'winner','lastiteration_winner','replay','row_commands','row_blind','row_budget','plan_sha',
    'bank_values','bank_late','winner_early','native_history','native_shift','native_terminal',
    'module_attempt','module_return','extra_AD','missing_packet','missing_return','candidate_durability',
])
def test_corruption_rejected_in_memory(saved,bad):
    t,slot,a,r,row,plan,events=deepcopy(saved[2][1])
    if bad=='seed':r['seed']+=1
    elif bad=='source_slot':r['source_slot']=1
    elif bad=='arm':r['arm']='random_shooting'
    elif bad=='taskid':r['task_id']='0'*64
    elif bad=='cost':r['candidates'][0]['cost']+=.01
    elif bad=='terminal':r['candidates'][0]['terminal'][0]+=.01
    elif bad=='candidate_logits':r['candidates'][0]['logits'][0][0]+=.01
    elif bad=='candidate_commands':r['candidates'][0]['commands'][0][0]+=.01
    elif bad=='candidate_index':r['candidates'][0]['index']=1
    elif bad=='candidate_iteration':r['candidates'][0]['iteration']=1
    elif bad=='missing_candidate':r['candidates'].pop()
    elif bad=='update_mean':r['updates'][0]['mean'][0][0]+=.01
    elif bad=='update_std':r['updates'][0]['std'][0][0]+=.01
    elif bad=='elite':r['updates'][0]['elite_indices'].reverse()
    elif bad=='winner':r['winner_index']=(r['winner_index']+1)%64
    elif bad=='lastiteration_winner':r['winner_index']=48
    elif bad=='replay':r['final_prediction']['segmented_replay'][-1]['action_history'][-1][0]+=.01
    elif bad=='row_commands':row['commands'][0][0]+=.01
    elif bad=='row_blind':row['blind_sha256']='0'*64
    elif bad=='row_budget':row['resources']['predict']['completed']=128
    elif bad=='plan_sha':plan['selection_sha256']='0'*64
    elif bad=='bank_values':next(e for e in events if e['kind']=='bank_frozen')['logits'][0][0][0]+=.01
    elif bad=='bank_late':
        i=next(i for i,e in enumerate(events) if e['kind']=='bank_frozen');events.append(events.pop(i))
    elif bad=='winner_early':
        i=next(i for i,e in enumerate(events) if e['kind']=='winner_frozen_before_replay');events.insert(1,events.pop(i))
    elif bad=='native_history':next(e for e in events if e['kind']=='native_packet')['context']['action_history'][-1][0]+=.01
    elif bad=='native_shift':next(e for e in events if e['kind']=='native_packet')['context']['latent_history'][0][0]+=.01
    elif bad=='native_terminal':next(e for e in events if e['kind']=='native_packet')['context']['latent_history'][-1][0]+=.01
    elif bad=='module_attempt':next(e for e in events if e['kind']=='operation_attempted')['event']['resource']='predict'
    elif bad=='module_return':next(e for e in events if e['kind']=='operation_completed')['event']['completed']=False
    elif bad=='extra_AD':events.insert(1,dict(kind='AD_API_attempt',entry=dict(resource='grad')))
    elif bad=='missing_packet':events.pop(next(i for i,e in enumerate(events) if e['kind']=='native_packet'))
    elif bad=='missing_return':events.pop(next(i for i,e in enumerate(events) if e['kind']=='operation_completed'))
    else:
        i=next(i for i,e in enumerate(events) if e['kind']=='candidate_saved');events.append(events.pop(i))
    with pytest.raises(ValueError):m.audit_branch(t,slot,a,r,row,plan,events)


@pytest.mark.parametrize('bad',['topk','iterations','candidatebudget','totalbudget','AD','backend','admission','equalclaim','floor'])
def test_protocol_mutation_rejected(saved,bad):
    c=deepcopy(saved[1]['contract'])
    if bad=='topk':c['CEM']['topk']=3
    elif bad=='iterations':c['CEM']['iterations']=5
    elif bad=='candidatebudget':c['candidate_predict']=134
    elif bad=='totalbudget':c['total_budget']['predict']=512
    elif bad=='AD':c['AD_API']=1
    elif bad=='backend':c['numerical']['SDP']='FLASH'
    elif bad=='admission':c['P1_admission']=True
    elif bad=='equalclaim':c['equal_D2P_compute_claim']=True
    else:c['reserve_bytes']=0
    with pytest.raises(ValueError):m.protocol(c)


def test_bad_external_report_sha():
    with pytest.raises(ValueError,match='SHA'):m.audit(report_sha='0'*64)


def test_bad_external_prepare_sha():
    with pytest.raises(ValueError):m.audit(prepare_sha='0'*64)


def test_no_producer_neural_or_physics_imports():
    # Examine the complete local import closure actually used by this audit.
    for mod in (m,m.pure,m.io):
        tree=ast.parse(Path(mod.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):names=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom):names=[node.module or '']+[x.name for x in node.names]
            else:continue
            assert all('torch' not in n and 'physics' not in n and not n.endswith('_cuda_20261027_v1') for n in names)


def test_duplicate_JSON_rejected():
    with pytest.raises(ValueError):m.io.parse(b'{"x":1,"x":2}')
