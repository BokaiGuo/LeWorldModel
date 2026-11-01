"""Shape-faithful RAM fixtures; no torch, training, simulator or CUDA calls."""
from copy import deepcopy
import json
import numpy as np
import pytest

from experiments.spectral_reachability import chat_p3_20261027_cuda as m
from experiments.spectral_reachability import chat_p3_20261027_scheduler as rule
from test_moving_history_budget_scheduler_cuda_20261027_v1 import RAM, Backend, task, banks, prepared


@pytest.fixture(scope='module')
def full():
    p=prepared(); ts=[task(s) for s in (0,1)]; bs=banks(); ram=RAM()
    meter=m.Meter(ram.event,m.old.fidelity.Limits())
    weights=[]
    for s in (0,1):
        j=np.zeros((192,20));j[:20]=np.eye(20)*(1+s)
        weights.append(m.old.signal(ts[s],np.asarray(ts[s]['root_context']['latent_history'][-1]),j)['w'])
    sealed=dict(weights=weights,plans={})
    for a in rule.ARMS:
        blocks=rule.allocation(a) if a in ('U','N') else rule.allocation(a,weights)
        sealed['plans'][a]=dict(blocks=blocks,jobs=rule.queue(blocks))
    result=m.census(p)
    m.panel(p,ts,bs,[Backend(s) for s in (0,1)],meter,result,sealed)
    return p,ts,bs,ram,meter,result,sealed


def test_actual_queue_and_saved_roundtrip(full):
    p,ts,bs,ram,meter,result,sealed=full
    assert result['allocations']['D_rank']!=result['allocations']['U']
    assert result['allocations']['S_rank']==result['allocations']['D_rank'][::-1]
    assert meter.counts()=={r:dict(attempted=v,completed=v) for r,v in m.old.MAXIMA.items()}
    # Sorted JSON roundtrip must not change replay dispatch order.
    result=json.loads(json.dumps(result,sort_keys=True))
    plans=m.validate_saved(p,ts,bs,result,ram.events,sealed)
    assert len(plans)==8
    assert all(not r['P1_admission'] and not r['P3_admission'] and not r['freshness'] for r in plans)
    axes=m.sample_axes(ram.events)
    assert axes['predictor']==dict(observed_calls=520,input_rows=520,context_tokens=1560)
    assert axes['pred_proj']==dict(observed_calls=520,input_rows=1560,context_tokens=None)
    for a in rule.ARMS:
        assert meter.counts(a)['predict']==dict(attempted=128,completed=128)


@pytest.mark.parametrize('fault', ['tail_cost','tail_raw','winner','replay','allocation','identity','summary'])
def test_saved_tampering_rejected(full,fault):
    p,ts,bs,ram,meter,result,sealed=full
    bad=deepcopy(result)
    tail=next(r for r in bad['rows'] if len(r['candidates'])==45)
    if fault=='tail_cost':tail['candidates'][44]['cost']+=1
    if fault=='tail_raw':tail['candidates'][44]['commands'][0][0]+=.01
    if fault=='winner':tail['winner_index']=44 if tail['winner_index']!=44 else 0
    if fault=='replay':tail['replays']['segmented_replay'][0]['latent_history'][-1][0]+=.01
    if fault=='allocation':bad['allocations']['D_rank']=[10,10]
    if fault=='identity':tail['task_id']='wrong'
    if fault=='summary':bad['paired_summary']['primary_mean']+=1
    with pytest.raises((ValueError,IndexError)):
        m.validate_saved(p,ts,bs,bad,ram.events,sealed)


@pytest.mark.parametrize('fault', ['api_return','module_output','all_receipts_history','double_module','api_timing'])
def test_ledger_and_actual_arrays_rejected(full,fault):
    p,ts,bs,ram,meter,result,sealed=full
    events=deepcopy(ram.events)
    if fault=='api_return':
        events.pop(next(i for i,e in enumerate(events) if e['kind']=='resource_return' and e['entry']['resource']=='predict'))
    if fault=='module_output':
        e=next(e for e in events if e['kind']=='module_output' and e['resource']=='pred_proj')
        a=m.old.unblob(e['arrays']);a[0,0]+=.1;e['arrays']=m.old.blob(a)
    if fault=='all_receipts_history':
        for e in events:
            if e['kind']=='context_input':e['context']['action_history'][0][0]+=1
    if fault=='double_module':events.append(deepcopy(next(e for e in events if e['kind']=='module_input')))
    if fault=='api_timing':
        e=next(e for e in events if e['kind']=='resource_return');e['entry']['ended_seconds']=-1
    with pytest.raises((ValueError,IndexError)):
        m.validate_saved(p,ts,bs,result,events,sealed)


def test_failure_keeps_denominator_and_no_retry(full):
    p,ts,bs,ram0,meter0,result0,sealed=full
    ram=RAM(); meter=m.Meter(ram.event,m.old.fidelity.Limits()); result=m.census(p)
    # Shared nominal+AD use4; fail on first candidate for slot0.
    with pytest.raises(ValueError,match='mock predict failure'):
        m.panel(p,ts,bs,[Backend(0,fail=5),Backend(1)],meter,result,sealed)
    assert len(result['rows'])==8 and meter.counts('U',0)['predict']==dict(attempted=1,completed=0)
    assert not any(r['candidates'] for r in result['rows'] if r['arm']!='U')
    assert meter.failed is False  # fixture raises outside meter.call; guard in real Native marks failed
    assert result['rows'][0]['candidates'][0]['status']=='attempted'


@pytest.mark.parametrize('scope,resource,cap', [(['U',0],'predict',64),(['D_rank',0],'predict',94),(['shared',0],'predict',4),(['shared',0],'svd',1)])
def test_own_scope_caps(scope,resource,cap):
    meter=m.Meter(lambda e:None,m.old.fidelity.Limits());meter.scope=scope
    for _ in range(cap):meter.call(resource,lambda:None)
    with pytest.raises(ValueError):meter.begin(resource)


def test_N_no_math_or_unbound_scope():
    meter=m.Meter(lambda e:None,m.old.fidelity.Limits());meter.scope=['N',0]
    with pytest.raises(ValueError):meter.begin('svd')
    meter.scope=['D_geometry',0]
    with pytest.raises(ValueError):meter.begin('predict')


def test_candidate_order_not_cost_sensitive(full):
    _,_,bs,_,_,result,sealed=full
    for arm in rule.ARMS:
        jobs=sealed['plans'][arm]['jobs']
        for s in (0,1):
            row=next(r for r in result['rows'] if r['arm']==arm and r['source_slot']==s)
            assert [c['index'] for c in row['candidates']]==[j['index'] for j in jobs if j['slot']==s]
            assert [c['commands'] for c in row['candidates']]==[bs[s][i]['commands'] for i in range(len(row['candidates']))]
