"""RAM dynamics/full-buffer fixtures; never a neural or physical experiment."""
from copy import deepcopy
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_r_native_orchestrator_20261029_v1 as m

fixtures=runpy.run_path(str(m.ROOT/'tests/test_agent_p3_parent_native_cuda_20261029_v1.py'))


class Memory:
    def __init__(self):self.events=[];self.capsules={}
    def event(self,e):self.events.append(deepcopy(e))
    def capsule(self,slot,cid,call):
        self.capsules[slot,cid]=deepcopy(call)
        return dict(path=f'/RAM/slot{slot}_call{cid}.json',sha256=m.digest(call))
    def reader(self,slot,cid,call):
        assert self.capsules[slot,cid]==call
        return dict(path=f'/RAM/slot{slot}_call{cid}.json',sha256=m.digest(call))


class NonlinearRAM(fixtures['NumpyNative']):
    """Explicit RAM x'=x-(1+10000y)*mean(raw), y'=y+.01*mean(raw).

    Original fixture's native entry/IO machinery retained; analytic raw-zero J
    is evaluated at EACH actual postcontext, not copied from root.
    """
    def call(self,name,fn,inputs,shape=None):
        if name=='action_encoder':self.outgoing=inputs['0'].copy()
        if name=='predictor':
            def transition():
                z=inputs['0'].copy();v=z[0,-1]
                token=self.outgoing[0,-1].reshape(5,2)
                raw=token*np.asarray(self.task['action_std'],np.float32)+np.asarray(self.task['action_mean'],np.float32)
                step=raw.mean(dtype=np.float32)
                v[0]-=np.float32(1+10000*v[1])*step;v[1]+=np.float32(.01)*step
                return z
            fn=transition
        return super().call(name,fn,inputs,shape)
    def signal(self,req,ledger):
        self.ledger=ledger;self.step=None;h=req['horizon'];zero=np.zeros((5*h,2),np.float32)
        self.phase='nominal';n=np.asarray(self.trajectory(req['parent_context'],zero,False)[-1]['latent_history'][-1],np.float32)
        self.arrays('nominal',value=n);self.phase='AD'
        def jacobian():
            ad=np.asarray(self.trajectory(req['parent_context'],zero,False)[-1]['latent_history'][-1],np.float32)
            self.step=None;self.arrays('AD_terminal',value=ad)
            j=np.zeros((192,10*h),np.float32);y=np.float32(req['parent_context']['latent_history'][-1][1])
            j[0]=-np.float32(1+10000*y)/np.float32(10);j[1]=np.float32(.001)
            for i,row in enumerate(j):
                self.sink.event(dict(kind='direction',scope=deepcopy(ledger.scope),direction=i))
                basis=np.eye(192,dtype=np.float32)[i]
                def raw_return():self.arrays('raw_backward',len(ledger.events)-1,value=row);return row.copy()
                ledger.call('vjp',lambda:self.call('grad',lambda:ledger.call('raw_backward',raw_return),
                    {'output':ad,'raw':zero.reshape(-1),'basis':basis}),directions=1)
            return j
        j=ledger.call('jacobian',jacobian);self.arrays('jacobian',value=j)
        return dict(binding_sha256=req['binding_sha256'],parent_context_sha256=req['parent_context_sha256'],horizon=h,
            nominal=n.tolist(),jacobian=j.tolist(),producer_sha256=m.sha(m.signal.__file__))


def r_task():
    t=fixtures['original']['task']();t['goal']=[1.,0.,0.,1.]+[0.]*188
    t['binding']['goal_sha256']=m.digest(t['goal'])
    return m.core.validate_task(t)


def one(mode='adaptive',*,caps=None,nonlinear=True):
    t=r_task() if nonlinear else fixtures['original']['task']()
    store=Memory();sink=type('Sink',(),{'event':lambda self,e:store.event(e)})()
    cls=NonlinearRAM if nonlinear else fixtures['NumpyNative']
    r=m.run_tree(t,lambda t,s:cls(t,s),sink,mode=mode,caps=caps,permit_ram=True,kind='RAM_FIXTURE',
        capsule=lambda cid,call:store.capsule(0,cid,call))
    return t,r,store


def test_original_caps_honest_R9_atomic_refusal_before_bank_or_control():
    t,r,s=one()
    assert r['status']=='resource_cap_before_node_queue' and r['model_plan'] is None
    q=r['queues'][0]
    assert q['decision']['extension_type']=='reconfiguration' and q['decision']['candidates']==8
    assert q['minimum']['replay_calls']==18 and q['verification_reserve']['replay_calls']==2
    assert r['resource_receipt']['caps']['replay_calls']==2
    assert r['R_groups']==[] and s.capsules=={} and len(r['native_calls'])==2
    assert all(x['status']=='not_requested' for x in r['jobs'])
    summary=m.validate_tree(t,r,s.events,m.parent.default_caps('adaptive'),permit_fixture=True,
        capsule_reader=lambda cid,call:s.reader(0,cid,call))
    assert summary['R_groups']==0 and summary['plan_available'] is False


@pytest.fixture(scope='module')
def executed_R():
    # RAM-only capacity override exercises dormant evaluator. No production cap changes.
    caps=m.parent.default_caps('adaptive');caps['replay_calls']=20
    return one(caps=caps)


def test_all9_actual_postcontexts_eligible_all_beam_and_native_IO(executed_R):
    t,r,s=executed_R
    assert r['status']=='tree_complete_not_admitted',r['errors']
    assert len(r['R_groups'])==1 and len(r['jobs'])==144
    g=r['R_groups'][0];assert g['status']=='witness_found'
    assert len(g['eligible_indices'])>1 and len(g['rows'])==8 and g['control']['status']=='complete'
    assert g['resources']==m.group_cost(1) and g['resources']['predict_API']==36
    assert g['resources']['array_SVD_calls']==10 # bank once + 9 actual ridge SVD
    assert g['bank']['resources']==m.spectral.bank_cost()
    eligible=[j['node']['indices'] for j in r['jobs'][:8] if j['status']=='returned_eligible']
    stopped=[q['parent']['indices'] for q in r['queues'][1:]]
    assert sorted(eligible)==sorted(stopped) # all eligible, not only reach winner
    assert len(r['replays'])==2 and all(x['status']=='returned' for x in r['replays'])
    assert len(r['model_plan']['actions'])==5 # suffix lookahead never appended
    assert r['resource_receipt']['attempted']['replay_calls']==20
    for row in g['rows']:
        assert row['signal']['parent_context_sha256']==m.digest(row['postcontext'])
        assert row['signal']['jacobian']!=g['control']['signal']['jacobian'] or row['postcontext']==g['control']['postcontext']
        assert len(row['nominal_path'][0]['latent_history'])==3
        assert len(row['nominal_path'][0]['action_history'])==2
    assert len(s.capsules)==1 and all(x is False for x in (r['P1_admission'],r['full_C3_demonstrated'],r['full_policy_executed']))


def test_complete_RAM_saved_algebra_including_timed_bank_hashes(executed_R):
    t,r,s=executed_R
    summary=m.validate_tree(t,r,s.events,r['resource_receipt']['caps'],permit_fixture=True,
        capsule_reader=lambda cid,call:s.reader(0,cid,call))
    assert summary['R_groups']==1 and summary['eligible_counts']==[len(r['R_groups'][0]['eligible_indices'])]
    assert summary['AD_primal_predict_API']>0 and summary['Jacobian_elements']>0


@pytest.mark.parametrize('damage',['unknown','missing','J','eligibility','onlywinner','raw','caps','eventcall'])
def test_saved_native_selection_mutations_refused(executed_R,damage):
    t,r,s=executed_R;v=deepcopy(r);es=deepcopy(s.events)
    if damage=='unknown':es.append(dict(kind='io',call=None,payload=dict(kind='unknown_event')))
    if damage=='missing':es.pop(next(i for i,e in enumerate(es) if e['payload']['kind']=='direction'))
    if damage=='J':v['R_groups'][0]['rows'][0]['signal']['jacobian'][0][0]+=1.
    if damage=='eligibility':v['R_groups'][0]['rows'][0]['eligible']=not v['R_groups'][0]['rows'][0]['eligible']
    if damage=='onlywinner':v['R_groups'][0]['eligible_indices']=[v['R_groups'][0]['diagnostic_reach_winner']]
    if damage=='raw':v['model_plan']['actions'][0][0]=.5
    if damage=='caps':v['resource_receipt']['caps']['replay_calls']+=1
    if damage=='eventcall':es[0]['call']=999
    with pytest.raises((ValueError,AssertionError)):
        m.validate_tree(t,v,es,r['resource_receipt']['caps'],permit_fixture=True,capsule_reader=lambda cid,call:s.reader(0,cid,call))


def test_fixed_progress_original_choose_and_caps_no_R():
    t,r,s=one(mode='fixed',nonlinear=False)
    assert r['status']=='tree_complete_not_admitted' and r['R_groups']==[]
    assert r['resource_receipt']['attempted']['predict_API']==114
    assert all(q['decision']['candidates']==4 and q['decision']['horizon']==1 for q in r['queues'])
    assert max(j['node']['depth'] for j in r['jobs'] if j['node'])==3
    m.validate_tree(t,r,s.events,m.parent.default_caps('fixed'),permit_fixture=True,capsule_reader=lambda cid,call:s.reader(0,cid,call))


def test_full_four_slot_execution_and_inert_validation():
    p=dict(tasks=[r_task(),r_task()],profile=m.profile());s=Memory()
    rows=m.execute(p,lambda t,s:NonlinearRAM(t,s),s,permit_ram=True)
    assert [r['status'] for r in rows]==['tree_complete_not_admitted','resource_cap_before_node_queue']*2
    summaries=m.validate_rows(p,rows,s.events,permit_fixture=True,capsule_reader=s.reader)
    assert len(summaries)==4 and all(len(r['result']['jobs'])==144 for r in rows)


@pytest.mark.parametrize('h',[1,3])
def test_group_cost_exact_h1_h3_bank_and_control_and_finalreserve(h):
    v=m.group_cost(h)
    assert v['predict_API']==9*(h+3) and v['replay_calls']==18
    assert v['jacobian_API']==9 and v['solve_calls']==9 and v['array_SVD_calls']==10
    assert v['program_compile_calls']==10
    node=m.parent.root_node(r_task());rr=m.reserve(node,h)
    assert rr['predict_API']==2*h and rr['replay_calls']==2


def test_no_witness_retains_all8_no_zero_fallback_and_no_final_replay():
    t=r_task();t['goal']=[0.]*191+[1.];t['binding']['goal_sha256']=m.digest(t['goal'])
    s=Memory();caps=m.parent.default_caps('adaptive');caps['replay_calls']=20
    r=m.run_tree(t,lambda t,s:NonlinearRAM(t,s),s,mode='adaptive',caps=caps,permit_ram=True,kind='RAM_FIXTURE',
        capsule=lambda cid,call:s.capsule(0,cid,call))
    assert r['status']=='no_witness' and not r['errors'] and r['model_plan'] is None
    g=r['R_groups'][0]
    assert g['status']=='no_witness' and g['eligible_indices']==[] and len(g['rows'])==8
    assert all(j['status']=='returned_ineligible' for j in r['jobs'][:8])
    assert r['resource_receipt']['attempted']['replay_calls']==18
    assert all(x['status']=='not_requested' for x in r['replays'])
    m.validate_tree(t,r,s.events,caps,permit_fixture=True,capsule_reader=lambda cid,call:s.reader(0,cid,call))


def test_native_runtime_failure_keeps_all_four_slots_and_fees():
    p=dict(tasks=[r_task(),r_task()],profile=m.profile());s=Memory()
    rows=m.execute(p,lambda t,s:fixtures['NumpyNative'](t,s,fail=True),s,permit_ram=True)
    assert rows[0]['status']=='failed_or_budget_exhausted'
    assert all(x['status']=='blocked_after_runtime_failure' for x in rows[1:])
    assert rows[0]['result']['resource_receipt']['attempted']['predict_API']==1
    m.validate_rows(p,rows,s.events,permit_fixture=True,capsule_reader=s.reader)


def test_no_implicit_RAM_in_saved_production(executed_R):
    t,r,s=executed_R
    with pytest.raises(ValueError,match='RAM production'):
        m.validate_tree(t,r,s.events,r['resource_receipt']['caps'],capsule_reader=lambda cid,call:s.reader(0,cid,call))


def test_run_authorization_and_cli_before_any_prepare_or_model():
    with pytest.raises(ValueError,match='MAIN source audit'):
        m.run('/not/read','bad','/not/written',authorization='worker')
    code='''
import sys
from experiments.spectral_reachability import agent_p3_r_native_orchestrator_20261029_v1 as m
assert 'torch' not in sys.modules and 'gymnasium' not in sys.modules
assert m.profile()['resource_caps']==m.old.profile()['resource_caps']
assert not m.rules()['full_policy_executed']
'''
    subprocess.run([sys.executable,'-c',code],check=True)
    subprocess.run([sys.executable,'-m',m.__name__,'--help'],check=True,capture_output=True)
