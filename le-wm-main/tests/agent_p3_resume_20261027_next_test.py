"""RAM/metadata only: never calls actual prepare, GPU, model or physics."""
from copy import deepcopy
import json
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_resume_20261027_next as m
from test_moving_history_budget_scheduler_cuda_20261027_v1 import RAM, Backend, task, banks, prepared


def fixture(weights=(2.,1.)):
    p=prepared(); original=banks(); bs=[m.cold_bank(s,original[s]) for s in (0,1)]
    p.update(weights=list(weights),banks=bs)
    p['queues']=m.queues(p['slots'],bs,p['weights'])
    return p,[task(s) for s in (0,1)],bs


@pytest.fixture(scope='module')
def full():
    p,ts,bs=fixture(); ram=RAM(); meter=m.Meter(ram.event,m.dependency().old.fidelity.Limits())
    result=m.census(p)
    m.panel(p,ts,bs,[Backend(s) for s in (0,1)],meter,result)
    return p,ts,bs,ram,meter,result


def test_bank_is_original_stream_plus_next_three_without_anchor_or_score_input():
    original=banks()
    for s in (0,1):
        b=m.cold_bank(s,original[s])
        assert len(b)==45 and len({m.digest(c['commands']) for c in b})==45
        for i in range(42):
            assert b[i]['commands']==original[s][i+3]['commands']
            assert b[i]['logits']==original[s][i+3]['logits']
        assert all(c['kind']=='PCG64_ball' and c['old_stream_index']==c['index']+3 for c in b)
        assert not any(c['commands']==a['commands'] for c in b for a in original[s][:3])
        # Existing scores do not enter bank generation.
        changed=deepcopy(original[s])
        for c in changed: c['cost']=-1.e9
        assert m.cold_bank(s,changed)==b
        changed[3]['commands'][0][0]=0
        with pytest.raises(ValueError): m.cold_bank(s,changed)


@pytest.mark.parametrize('w',[(2.,1.),(1.,2.),(1.,1.),(0.,0.)])
def test_fixed_equal_budget_floor_tie_and_literal_requests(w):
    p,_,bs=fixture(w)
    for a,q in p['queues'].items():
        assert sum(q['blocks'])==20 and min(q['blocks'])>=5 and len(q['requests'])==60
        assert sum(j['predict_API'] for j in q['requests'])==120
        assert [j['slot'] for j in q['requests'][:30]]==[s for _ in range(5) for s in (0,0,0,1,1,1)]
        for s in (0,1):
            req=[j for j in q['requests'] if j['slot']==s]
            assert [j['index'] for j in req]==list(range(3*q['blocks'][s]))
            for j in req:
                assert j['candidate_sha256']==m.digest(bs[s][j['index']])
                assert j['commands']==bs[s][j['index']]['commands']
                assert j['identity']['task_id']==p['slots'][s]['task_id']
    assert p['queues']['U']=={**p['queues']['N'],'requests':[{**j,'arm':'U'} for j in p['queues']['N']['requests']]}
    if w[0]==w[1]: assert all(q['blocks']==[10,10] for q in p['queues'].values())
    else: assert len({tuple(q['blocks']) for q in p['queues'].values()})==3


def test_full_RAM_execution_saved_bridge_equal_resource_and_no_science(full):
    p,ts,bs,ram,meter,result=full
    actual=json.loads(json.dumps(result,sort_keys=True))
    ps=m.validate_saved(p,ts,bs,actual,ram.events)
    assert len(ps)==8 and all(x[k] is False for x in ps for k in m.FLAGS)
    assert meter.counts()=={k:dict(attempted=v,completed=v) for k,v in m.maxima().items()}
    for a in m.ARMS:
        assert meter.counts(a)['predict']==dict(attempted=128,completed=128)
        ax=m.dependency().sample_axes([e for e in ram.events if e['scope'][0]==a])
        assert ax['predictor']==dict(observed_calls=128,input_rows=128,context_tokens=384)
    assert m.dependency().sample_axes(ram.events)['pred_proj']['input_rows']==1536
    assert m.protocol()['equal_incremental_resource_vector'] is True
    assert m.protocol()['equal_wall'] is m.protocol()['equal_FLOPs'] is False


@pytest.mark.parametrize('fault',['missing_branch','extra_candidate','tail_cost','tail_raw','winner','plan_identity',
    'plan_flag','replay','allocation','task','summary','paid_bool','missing_API','double_module','wrong_module',
    'input_context','entry_bool','time','native_packet','packet_chronology'])
def test_mutation_rejection(full,fault):
    p,ts,bs,ram,_,result=full; bad=deepcopy(result); ev=deepcopy(ram.events)
    tail=next(r for r in bad['rows'] if len(r['candidates'])==45)
    if fault=='missing_branch': bad['rows'].pop()
    elif fault=='extra_candidate': tail['candidates'].append(deepcopy(tail['candidates'][0]))
    elif fault=='tail_cost': tail['candidates'][-1]['cost']+=.1
    elif fault=='tail_raw': tail['candidates'][-1]['commands'][0][0]=0
    elif fault=='winner': tail['winner_index']=-1
    elif fault=='plan_identity': bad['rows'][0]['plan']['source_slot']=False
    elif fault=='plan_flag': tail['plan']['P1_admission']=True
    elif fault=='replay': tail['replays']['whole_replay'][0]['latent_history'][-1][0]+=.1
    elif fault=='allocation': bad['allocations']['D_rank']=[10,10]
    elif fault=='task': tail['task_id']='other'
    elif fault=='summary': bad['paired_summary']['primary_mean']+=1
    elif fault=='paid_bool': next(e for e in ev if e['kind']=='candidate_request')['paid']=True
    elif fault=='missing_API': ev.pop(next(i for i,e in enumerate(ev) if e['kind']=='resource_return'))
    elif fault=='double_module': ev.append(deepcopy(next(e for e in ev if e['kind']=='module_input')))
    elif fault=='wrong_module':
        e=next(e for e in ev if e['kind']=='module_output' and e['resource']=='pred_proj')
        a=m.dependency().old.unblob(e['arrays']); a[0,0]+=.1; e['arrays']=m.dependency().old.blob(a)
    elif fault=='input_context': next(e for e in ev if e['kind']=='context_input')['context']['action_history'][0][0]+=1
    elif fault=='entry_bool': next(e for e in ev if e['kind']=='resource_attempt')['entry']['index']=False
    elif fault=='time': next(e for e in ev if e['kind']=='resource_return')['entry']['ended_seconds']=-1
    elif fault=='native_packet': next(e for e in ev if e['kind']=='native_packet')['context']['latent_history'][-1][0]+=1
    elif fault=='packet_chronology':
        index=next(i for i,e in enumerate(ev) if e['kind']=='native_packet')
        ev.insert(0,ev.pop(index))
    with pytest.raises((ValueError,IndexError,KeyError)):
        m.validate_saved(p,ts,bs,bad,ev)


def test_failure_preserves_full_denominator_no_retry():
    p,ts,bs=fixture(); ram=RAM(); meter=m.Meter(ram.event,m.dependency().old.fidelity.Limits()); result=m.census(p)
    with pytest.raises(ValueError,match='mock predict failure'):
        m.panel(p,ts,bs,[Backend(0,fail=1),Backend(1)],meter,result)
    assert len(result['rows'])==8 and result['rows'][0]['status']=='attempted'
    assert meter.counts('U',0)['predict']==dict(attempted=1,completed=0)
    assert sum(len(r['candidates']) for r in result['rows'])==1


def test_no_search_AD_or_shared_reacquisition():
    meter=m.Meter(lambda e:None,m.dependency().old.fidelity.Limits())
    for scope,resource in [(['U',0],'svd'),(['N',0],'grad_api'),(['shared',0],'predict')]:
        meter.scope=scope
        with pytest.raises(ValueError): meter.begin(resource)


def test_external_prepare_SHA_and_whole_material_rebuild(monkeypatch):
    p,ts,bs=fixture(); seal='a'*64
    monkeypatch.setattr(m,'material',lambda:(p,ts,bs))
    monkeypatch.setattr(m,'sha',lambda _:seal)
    monkeypatch.setattr(m.prior,'read',lambda _:deepcopy(p))
    assert m.verify('/RAM/never_read',seal)[0]==p
    with pytest.raises(ValueError): m.verify('/RAM/never_read','b'*64)
    bad=deepcopy(p); bad['queues']['D_rank']['requests'].reverse()
    monkeypatch.setattr(m.prior,'read',lambda _:bad)
    with pytest.raises(ValueError): m.verify('/RAM/never_read',seal)


def test_run_requires_MAIN_before_files_or_imports():
    with pytest.raises(ValueError,match='MAIN audit'):
        m.run(None,None,None,None)
    assert not any(k in sys.modules for k in ('torch','gym','pymunk'))


def test_exclusive_new_prefix_not_old_or_shared():
    for p in (m.ROOT/'results/chat_p3_20261027_old.json',m.ROOT/'innerdocs/agent_p3_resume_20261027_next_bad.json'):
        with pytest.raises(ValueError): m.exclusive(p)


def test_model_differences_keep_zero_and_negative(full):
    bad=deepcopy(full[-1])
    for r in bad['rows']: r['plan']['model_cost']=1.
    next(r for r in bad['rows'] if r['arm']=='D_rank' and r['source_slot']==1)['plan']['model_cost']=0.
    assert m.paired(bad)['D_minus_control']['U']==[0.,-1.]


def test_diagnosis_distinguishes_anchor_saturation_from_dead_candidate_stream():
    original=banks(); p,_,_=fixture(); rows=[]
    for a in m.ARMS:
        for s in (0,1):
            count=3*p['queues'][a]['blocks'][s]; cs=[]
            for i in range(count):
                cs.append(dict(original[s][i],cost=float(i-3),
                    contexts=[dict(latent_history=[[float(i)]*192])]))
            rows.append(dict(arm=a,source_slot=s,candidates=cs,winner_index=0,plan=dict(model_cost=-3.)))
    result=dict(rows=rows,allocations={a:p['queues'][a]['blocks'] for a in m.ARMS})
    d=m.diagnose_rows(result,original)
    assert d['all_winners_in_common_first_three'] and d['quota_separated']
    assert all(r['best_proposal_minus_winner']==3. for r in d['rows'])
    assert all(r['distinct_proposal_commands']==r['distinct_proposal_predicted_endpoints']==r['paid_candidates']-3 for r in d['rows'])
    assert d['physical_reachability'] is None and d['physical_outcomes_read'] is False
    result['rows'][2]['candidates'][3]['cost']+=1
    with pytest.raises(ValueError,match='repeated same-root'):
        m.diagnose_rows(result,original)
