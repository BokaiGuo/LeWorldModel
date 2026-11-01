"""Independent arithmetic auditor: pure array fixtures, no actual model calls."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import p1_two_stage_witness_saved_audit_20261027_v1 as audit
from experiments.spectral_reachability import p1_two_stage_witness_20261027_v1 as producer

# Reuse ONLY the sealed NumPy mock backend/task fixture; NEVER the CUDA backend.
spec=importlib.util.spec_from_file_location('sealed_witness_array_mock',
    Path(__file__).with_name('test_p1_two_stage_witness_20261027_v1.py'))
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


class ArrayReceipts(fixture.ArrayBackend):
    def packet(self,t,c,logits,meter,label):
        after=super().packet(t,c,logits,meter,label)
        if label!='local_AD':
            meter.emit('packet_completed',label=label,input_context_sha256=audit.core.digest(c),
                logits=np.asarray(logits).tolist(),commands=np.tanh(np.asarray(logits,np.float32)).reshape(5,2).tolist(),context=after)
        return after

    def rollout(self,t,raw,meter,mode):
        path=super().rollout(t,raw,meter,mode)
        if mode!='root_AD':
            for k,c in enumerate(path):meter.emit('native_packet',mode=mode,packet=k,context=c)
        return path

    def jacobian(self,t,inc,meter):
        z,j=super().jacobian(t,inc,meter)
        meter.emit('jacobian_forward_complete',terminal=z,logits=inc['logits'])
        for i,row in enumerate(j):meter.emit('jacobian_row',row=i,raw_control_derivative=row.tolist())
        meter.emit('joint_logit_jacobian',terminal=z,jacobian=j.tolist())
        return z,j

    def suffix_jacobian(self,t,c,nominal,meter):
        d=super().suffix_jacobian(t,c,nominal,meter)
        for i,row in enumerate(d['jacobian']):meter.emit('suffix_J_row',row=i,derivative=row)
        meter.emit('suffix_J_complete',derivative=d)
        return d


@pytest.fixture(scope='module')
def arrays():
    task=fixture.task();events=[]
    result=producer.run_task(task,ArrayReceipts(),events.append)
    assert result['status']=='witness_completed_not_admitted'
    return task,result,events


def test_independent_all_slot_fullH2_and_budget_audit(arrays):
    result=audit.check_root(*arrays)
    assert result['status']=='complete_saved_arrays_consistent' and result['complete_H2']
    assert result['completed_candidates']==64 and result['unique_prefix_contexts']>1
    assert result['unique_suffix_J_arrays']>1 and result['suffix_J_pairwise_frobenius']['max']>0
    assert result['resources']['producer']['predict']['completed']==120
    assert result['resources']['witness']['predict']['completed']==6
    assert not result['P1_admission'] and not result['model_task_eligible'] and not result['direct_plateau_established']


@pytest.mark.parametrize('damage',['q','goal','cost','actions','mid_history','partialH2','winner','replay',
    'task','J_context','J_rows','J_duplicate_row','AD_context','prefix_bank','suffix_logits',
    'svd','solve','cpu_cost','phase','double_completion','missing_attempt','reported_cost','stage_shift',
    'missing_native_replay','budget','eligibility','stall'])
def test_failclosed_corruption(arrays,damage):
    t,r,e=deepcopy(arrays)
    if damage=='q':t['projection'][0][0]=0
    elif damage=='goal':t['goal'][0]+=.1
    elif damage=='cost':r['candidates'][1]['cost']+=.01
    elif damage=='actions':r['candidates'][1]['commands'][0][0]+=.01
    elif damage=='mid_history':r['candidates'][0]['contexts'][0]['action_history'][0][0]+=.01
    elif damage=='partialH2':r['candidates'][0]['contexts'].pop()
    elif damage=='winner':r['winner']=deepcopy(r['candidates'][-1])
    elif damage=='replay':r['witness']['segmented_replay'][1]['latent_history'][-1][0]+=.1
    elif damage=='task':r['task_id']='0'*64
    elif damage=='J_context':r['prefix_compilations'][1]['derivative']['input_context_sha256']='0'*64
    elif damage=='J_rows':next(v for v in e if v['kind']=='suffix_J_row')['derivative'][0]+=.1
    elif damage=='J_duplicate_row':next(v for v in e if v['kind']=='suffix_J_row')['row']=1
    elif damage=='AD_context':next(v for v in e if v['kind']=='suffix_AD_forward')['input_context_sha256']='0'*64
    elif damage=='prefix_bank':next(v for v in e if v['kind']=='prefix_slots_frozen')['logits'][4][0]+=.1
    elif damage=='suffix_logits':r['candidates'][2]['logits'][1][0]+=.1
    elif damage=='svd':r['prefix_compilations'][1]['compilation']['singular_values'][0]+=.1
    elif damage=='solve':r['prefix_compilations'][1]['compilation']['step'][0]+=.1
    elif damage=='cpu_cost':r['prefix_compilations'][1]['compilation']['arithmetic']['coefficient_evaluations']+=1
    elif damage=='phase':next(v for v in e if v['kind']=='operation_attempted')['event']['phase']='witness'
    elif damage=='double_completion':e.append(deepcopy(next(v for v in e if v['kind']=='operation_completed')))
    elif damage=='missing_attempt':e.pop(next(i for i,v in enumerate(e) if v['kind']=='operation_attempted'))
    elif damage=='reported_cost':r['resources']['producer']['predict']['completed']-=1
    elif damage=='stage_shift':next(v for v in e if v['kind']=='operation_completed')['stage']='prefix_99'
    elif damage=='missing_native_replay':e.pop(next(i for i,v in enumerate(e) if v['kind']=='native_packet'))
    elif damage=='budget':r['budget']['producer']['predict']=119
    elif damage=='eligibility':r['model_task_eligible']=True
    else:r['direct_plateau_established_by_this_producer']=True
    with pytest.raises(ValueError):audit.check_root(t,r,e)


def test_identical_J_not_a_failure_if_context_protocol_is_legal():
    class ConstantJ(ArrayReceipts):
        def __init__(self):super().__init__();self.b[:]=0
    t=fixture.task();events=[];r=producer.run_task(t,ConstantJ(),events.append)
    a=audit.check_root(t,r,events)
    assert a['unique_suffix_J_arrays']==1 and a['suffix_J_pairwise_frobenius']['max']==0
    assert a['completed_candidates']==64 and not a['P1_admission']


def test_partial_root_preserves_failure_and_no_fullH2_claim():
    events=[];t=fixture.task();r=producer.run_task(t,ArrayReceipts('after_local_AD'),events.append)
    a=audit.check_root(t,r,events)
    assert a['status']=='partial_saved_evidence' and not a['complete_H2']
    assert a['root_error']['message']=='local_AD_primary'
    assert a['resources']['producer']['jacobian']==dict(attempted=2,completed=1)


def test_out_of_order_operation_completions_rejected(arrays):
    t,r,e=deepcopy(arrays)
    a=next(i for i,x in enumerate(e) if x['kind']=='operation_attempted')
    b=next(i for i,x in enumerate(e) if x['kind']=='operation_completed')
    e[a],e[b]=e[b],e[a]
    with pytest.raises(ValueError):audit.check_root(t,r,e)


def test_budget_independent_of_producer_helper(monkeypatch):
    monkeypatch.setattr(producer,'budgets',lambda:pytest.fail('must not call budget helper'))
    assert audit.EXPECTED['producer']['predict']==120
    assert sum(audit.EXPECTED[p]['predict'] for p in audit.EXPECTED)*2==252
