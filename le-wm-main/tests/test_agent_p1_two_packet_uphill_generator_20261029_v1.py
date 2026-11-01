from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p1_two_packet_uphill_generator_20261029_v1 as m
import test_agent_nonlinear_suffix_observed_recompile_20261028_v1 as fixture

inputs=fixture.inputs
complete=fixture.complete


@pytest.fixture
def requests(complete):
    old,_,_=fixture.make_request(complete)
    task=old['task'];x=np.zeros(10);raw=np.concatenate([m.F.chart(x),m.F.chart(x)])
    return [dict(slot=i,root_slot=i//2,condition=('off','on')[i%2],task=deepcopy(task),
                 winner_prefix_logits=x.tolist(),winner_suffix_logits=x.tolist(),winner_commands=raw.tolist(),
                 winner_raw_bits=m.F.bits(raw),winner_authority='a'*64) for i in range(4)]


def test_paired_actual_callbacks_equal_cost_and_same_baseline(requests):
    q=requests[0];backends=[];results=[]
    for arm in m.ARMS:
        b=fixture.Arithmetic(q['task']);events=[];r=m.run_arm(q,arm,b,events.append)
        assert len(b.calls)==36 and sum(x['kind']=='ADforward' for x in b.calls)==16
        assert len(r['iterates'])==9 and len(r['gradients'])==8
        assert r['iterates'][0]['raw_bits']==q['winner_raw_bits']
        assert all(v['witness']==(v['stage_Q']>r['initial_Q'] and v['terminal_Q']<r['original_winner_terminal_Q']) for v in r['iterates'])
        backends.append(b);results.append(r)
    assert results[0]['iterates'][0]==results[1]['iterates'][0]
    assert results[0]['iterates'][1]['prefix_logits']!=results[1]['iterates'][1]['prefix_logits']
    assert results[0]['iterates'][1]['suffix_logits']==results[1]['iterates'][1]['suffix_logits']


def test_all_four_two_arms_and_fixed_budget(requests):
    events=[];p=m.panel(requests,lambda q,a:fixture.Arithmetic(q['task']),events.append)
    assert p['status']=='complete_model_only' and len(p['rows'])==4 and p['candidate_denominator']==72
    assert all(e['status']=='sealed' for row in p['rows'] for e in row['arms'])
    assert m.contract()['total']==dict(predict=288,scalar_reverse=128)
    assert not p['physical_release'] and not p['P1_admission']


def test_failure_keeps_all_slots_and_no_fallback(requests):
    calls=[]
    def factory(q,a):
        calls.append((q['slot'],a))
        if q['slot']==1 and a=='stage_ascent':raise RuntimeError('injected')
        return fixture.Arithmetic(q['task'])
    p=m.panel(requests,factory,lambda e:None)
    assert p['status']=='partial_failclosed'
    assert [[a['status'] for a in row['arms']] for row in p['rows']]==[
        ['sealed','sealed'],['sealed','failed_partial'],['not_started','not_started'],['not_started','not_started']]
    assert calls[-1]==(1,'stage_ascent') and p['rows'][1]['arms'][1]['result'] is None


@pytest.mark.parametrize('mutate',['physics','bad_bits','slot','nan'])
def test_request_rejects_outcome_and_bad_identity(requests,mutate):
    q=deepcopy(requests[0])
    if mutate=='physics':q['reward']=1.
    elif mutate=='bad_bits':q['winner_raw_bits']='00'
    elif mutate=='slot':q['root_slot']=1
    else:q['winner_prefix_logits'][0]=float('nan')
    with pytest.raises(ValueError):m.request(q)


def test_bad_ad_context_cannot_publish(requests):
    class Bad(fixture.Arithmetic):
        def gradient(self,c,a,label):
            d=super().gradient(c,a,label);d['context']['latent_history'][-1][0]+=1
            return d
    p=m.panel(requests,lambda q,a:Bad(q['task']),lambda e:None)
    assert p['rows'][0]['arms'][0]['status']=='failed_partial'
    assert p['rows'][0]['arms'][0]['result'] is None
