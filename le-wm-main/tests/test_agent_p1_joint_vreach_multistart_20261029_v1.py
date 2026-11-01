"""Synthetic arithmetic callbacks: no neural inference or real authority."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p1_joint_vreach_multistart_20261029_v1 as m
import test_agent_p1_two_packet_uphill_generator_20261029_v1 as fixture
import test_agent_nonlinear_suffix_observed_recompile_20261028_v1 as arithmetic

inputs=fixture.inputs
complete=fixture.complete
requests=fixture.requests


class Backend(arithmetic.Arithmetic):
    def signal(self,c,label):
        self.calls.append(dict(kind='signal',label=label,context=deepcopy(c)))
        return dict(nominal_context=self.after(c,np.zeros((5,2),np.float32)),jacobian=self.j.tolist())


def test_two_arms_fixed_budget_vreach_and_action_replay(requests):
    out=[]
    for arm in m.ARMS:
        b=Backend(requests[0]['task']);events=[];r=m.run_arm(requests[0],arm,b,events.append);out.append(r)
        assert len(r['candidates'])==20 and r['resources']['attempted']==r['resources']['completed']==m.contract()['callback_cost_per_arm']
        assert sum(c['kind']=='signal' for c in b.calls)==20 and sum(c['kind']=='ADforward' for c in b.calls)==16
        for v in r['candidates']:
            assert v['witness']==(v['stage_uphill'] and v['reach_better'] and v['terminal_better'])
            assert m.F.bits(v['commands'])==v['raw_float32_bits']
            assert v['own_reach_value']==v['suffix_solve']['own_reach_value']
        assert not r['P1_admission'] and not r['physical_release']
    assert out[0]['baseline_terminal_Q']==out[1]['baseline_terminal_Q']
    assert out[0]['baseline_reach_value']==out[1]['baseline_reach_value']
    assert m.contract()['candidate_denominator']==160 and m.contract()['callback_cost_total']['predict']==1104


def test_tangent_removes_only_outward_component():
    x=np.array([1.,0.]+[0.]*8)
    assert np.allclose(m.tangent_ascent(x,[2.,3.]+[0.]*8),[0.,3.]+[0.]*8)
    assert np.allclose(m.tangent_ascent(x,[-2.,3.]+[0.]*8),[-2.,3.]+[0.]*8)
    assert np.allclose(m.tangent_ascent(np.zeros(10),[2.,3.]+[0.]*8),[2.,3.]+[0.]*8)


def test_full_panel_and_fixed_roster(requests):
    p=m.panel(requests,lambda q,a:Backend(q['task']),lambda e:None)
    assert p['status']=='complete_model_only' and p['arm_denominator']==8 and p['candidate_denominator']==160
    assert all(a['status']=='sealed' for row in p['rows'] for a in row['arms'])


def test_failed_callback_preserves_attempts_and_no_fallback(requests):
    class Bad(Backend):
        def signal(self,c,label):raise RuntimeError('signal failure')
    p=m.panel(requests,lambda q,a:Bad(q['task']),lambda e:None)
    entry=p['rows'][0]['arms'][0];snap=entry['failed_resource_snapshot']
    assert entry['status']=='failed_partial' and entry['result'] is None
    assert snap['attempted']['jacobian']==1 and snap['completed']['jacobian']==0
    assert p['rows'][0]['arms'][1]['status']=='not_started' and p['candidate_denominator']==160


def test_bad_signal_context_or_outcome_rejected(requests):
    class Bad(Backend):
        def signal(self,c,label):
            x=super().signal(c,label);x['nominal_context']['action_history'][-1][0]+=1;return x
    with pytest.raises(ValueError):m.run_arm(requests[0],m.ARMS[0],Bad(requests[0]['task']),lambda e:None)
    q=deepcopy(requests[0]);q['reward']=1
    with pytest.raises(ValueError):m.run_arm(q,m.ARMS[0],Backend(q['task']),lambda e:None)


def test_mismatched_paid_baseline_refuses_second_arm(requests):
    def factory(q,arm):
        b=Backend(q['task'])
        if arm==m.ARMS[1]:b.j*=2
        return b
    p=m.panel(requests,factory,lambda e:None)
    assert p['status']=='partial_failclosed' and p['rows'][0]['arms'][1]['status']=='failed_partial'
    assert p['rows'][0]['arms'][1]['result'] is None and 'same paid baseline' in p['rows'][0]['arms'][1]['error']
