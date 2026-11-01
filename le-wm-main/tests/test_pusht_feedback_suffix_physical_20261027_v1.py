"""Non-neural contract and injected-env tests; never real physics or CUDA."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from experiments.spectral_reachability import pusht_feedback_suffix_physical_20261027_v1 as m
from experiments.spectral_reachability import pusht_full_packet_compiler_20261027 as packet


def rgb(step):
    a = np.zeros((224,224,3),dtype=np.uint8)
    a[0,0] = (65,105,225); a[2,2] = step
    return a


@pytest.fixture
def inputs(tmp_path):
    """Explicit algebra fixture, not a simulated neural efficacy result."""
    q=np.eye(192);goal=np.ones(192)*.1;jac=np.zeros((192,10));jac[:10]=np.diag(np.arange(1,11))
    nominal=np.zeros(192);bank=packet.spectral_bank(jac,nominal,goal,q)
    cases=[];contexts=[];saved=[];records=[]
    image=tmp_path/'prefix.png';Image.fromarray(rgb(5)).save(image)
    for role in m.ROLES:
        states={}
        for arm in m.ARMS:
            context=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
                latent_history=np.zeros((3,192)).tolist(),action_history=np.zeros((2,10)).tolist())
            states[arm]=context;prefix=np.zeros((5,2)).tolist();candidates=[]
            for i,u in enumerate(bank['params']):
                u=np.asarray(u,np.float32).astype(float);terminal=jac@np.tanh(u)
                candidates.append(dict(candidate_id=f'suffix_{i}',kind=bank['kinds'][i],suffix_params=u.tolist(),
                    commands=prefix+np.tanh(u).reshape(5,2).tolist(),predicted_terminal=terminal.tolist(),
                    nonlinear_terminal_cost=float(np.mean((q@(goal-terminal))**2))))
            saved.append(dict(root_sha256=m.ROOT,root_index=1,role=role,arm=arm,prefix=prefix,context=context,
                q=q.tolist(),goal=goal.tolist(),nominal_terminal=nominal.tolist(),jacobian=jac.tolist(),bank=bank,
                candidates=candidates,selected_candidate_id=min(candidates,key=lambda c:c['nonlinear_terminal_cost'])['candidate_id'],
                independent_replay_exact=True))
            for suffix,n in (('_compile',2),('_nonlinear_bank',10),('_independent_replay',1)):
                records.append(dict(method=role+'_'+arm+suffix,predict_calls=n,predict_samples=n,completed=True,
                    failed_predict_calls=0,horizon_tokens=1,context_lengths={'3':n},
                    phase='evaluation' if suffix=='_independent_replay' else 'planning'))
        cases.append(dict(role=role,prefix=prefix,prefix_rgb=str(image)))
        contexts.append(dict(role=role,**states))
    report=dict(schema=m.COMPILER,status='plans_sealed',failure=None,model_unchanged=True,
        partial_internal_work_unknown=False,actual_predict_samples=78,jacobian_attempts=6,jacobian_materializations=6,
        scalar_vjp_directions_completed=1152,svd_calls_completed=12,new_encode_calls=0,new_physics_steps=0,
        plans=saved,predict=records)
    protocol=dict(cases=cases,q=q.tolist(),goal=goal.tolist())
    feedback=dict(status='diagnostic_recorded',failure=None,contexts=contexts)
    return report,protocol,feedback


def test_six_exact_saved_winners_and_aliases(inputs):
    plans=m.plans_from_saved(*inputs)
    assert len(plans)==6 and len({p['method'] for p in plans})==6
    assert [(p['role'],p['arm']) for p in plans]==[(r,a) for r in m.ROLES for a in m.ARMS]
    assert m.census(plans)['planned_selected_path_steps']==60
    for p,s in zip(plans,inputs[0]['plans']):
        assert p['commands']==next(c['commands'] for c in s['candidates'] if c['candidate_id']==s['selected_candidate_id'])
        assert p['corrected_search_horizon']==1 and p['planned_horizon_tokens']==2


@pytest.mark.parametrize('mutation',['duplicate','root','root_bool','prefix','context','q','winner','cost','commands',
    'bank','replay','calls','scope_order','samples','jac','partial','status'])
def test_saved_plan_tampering_rejected(inputs,mutation):
    r,p,f=map(deepcopy,inputs);s=r['plans'][0]
    if mutation=='duplicate':r['plans'][1]=deepcopy(s)
    elif mutation=='root':s['root_sha256']='x'*64
    elif mutation=='root_bool':s['root_index']=True
    elif mutation=='prefix':s['prefix'][0][0]=.01
    elif mutation=='context':s['context']['latent_history'][0][0]=.001
    elif mutation=='q':s['q'][0][0]=2
    elif mutation=='winner':s['selected_candidate_id']='suffix_0'
    elif mutation=='cost':s['candidates'][0]['nonlinear_terminal_cost']+=.001
    elif mutation=='commands':s['candidates'][0]['commands'][6][0]=.1
    elif mutation=='bank':s['candidates'][2]['suffix_params'][0]+=.01
    elif mutation=='replay':s['independent_replay_exact']=False
    elif mutation=='calls':r['predict'][0]['predict_calls']=1
    elif mutation=='scope_order':r['predict'][0],r['predict'][1]=r['predict'][1],r['predict'][0]
    elif mutation=='samples':r['actual_predict_samples']=77
    elif mutation=='jac':r['jacobian_materializations']=5
    elif mutation=='partial':r['partial_internal_work_unknown']=True
    else:r['status']='integrity_fail'
    with pytest.raises((ValueError,KeyError)):
        m.plans_from_saved(r,p,f)


def test_selection_receipt_and_census_tamper(inputs):
    plans=m.plans_from_saved(*inputs);plans[0]['commands'][7][0]=.2
    with pytest.raises(ValueError,match='receipt'):
        m.census(plans)


class FakeEnv:
    def __init__(self,terminate=None,step_error=None,render_error=None,visible=True):
        self.n=0;self.terminate=terminate;self.step_error=step_error;self.render_error=render_error
        self.visible=visible;self.actions=[]
    def render(self):
        if self.n==self.render_error:raise RuntimeError('injected render failure')
        a=rgb(self.n)
        if not self.visible:a[0,0]=0
        return a
    def step(self,command):
        self.actions.append(command.copy())
        if len(self.actions)==self.step_error:raise RuntimeError('injected step failure')
        self.n+=1
        return None,0.,self.n==self.terminate,False,{}


def execute(env,expected=None):
    events=[]
    result=m.execute_ten(env,np.zeros((10,2)),root_id=m.ROOT,expected_prefix=rgb(5) if expected is None else expected,
        state_of=lambda e:np.array([e.n]*7),event=lambda e,image:events.append((deepcopy(e),None if image is None else image.copy())))
    return result,events


def test_exact_five_plus_five_order_and_prefix_gate():
    env=FakeEnv();r,events=execute(env)
    assert r['status']=='completed' and r['full_horizon_completed']
    assert r['physical_env_step_attempts']==r['physical_env_step_returns']==10
    assert r['prefix_identity']['pixels_exact'] is True
    assert [e[0]['primitive_step'] for e in events if e[0]['kind']=='attempt']==list(range(1,11))
    kinds=[(e[0]['kind'],e[0].get('primitive_step')) for e in events]
    assert kinds.index(('observation',5))<kinds.index(('prefix_identity',5))<kinds.index(('attempt',6))


def test_prefix_mismatch_stops_before_sixth_dispatch_with_step5_retained():
    expected=rgb(5);expected[3,3]=123;env=FakeEnv();r,events=execute(env,expected)
    assert r['status']=='prefix_mismatch' and len(env.actions)==5
    assert r['completed_commands']==r['attempted_commands']==len(r['trace'])==5
    assert r['remaining_commands_not_executed']==5 and not r['full_horizon_completed']
    assert [e[0]['primitive_step'] for e in events if e[0]['kind']=='observation']==[1,2,3,4,5]


def test_early_termination_never_substitutes_terminal():
    r,_=execute(FakeEnv(terminate=4))
    assert r['status']=='environment_terminated' and r['completed_commands']==4
    assert r['prefix_identity']['status']=='not_reached' and not r['full_horizon_completed']


def test_termination_at_prefix_never_executes_suffix():
    env=FakeEnv(terminate=5);r,_=execute(env)
    assert r['prefix_identity']['status']=='matched' and len(env.actions)==5
    assert r['status']=='environment_terminated' and not r['full_horizon_completed']


def test_step_exception_accounts_attempt_before_return():
    r,_=execute(FakeEnv(step_error=3))
    assert r['status']=='execution_failed'
    assert r['physical_env_step_attempts']==3 and r['physical_env_step_returns']==2


def test_render_exception_charges_return_but_no_invented_observation():
    r,_=execute(FakeEnv(render_error=3))
    assert r['completed_commands']==3 and len(r['trace'])==2
    assert r['status']=='execution_failed'


def test_guard_no_visibility_zero_dispatch():
    env=FakeEnv(visible=False);r,_=execute(env)
    assert r['status']=='observation_unavailable' and len(env.actions)==0
    assert r['completed_commands']==r['attempted_commands']==0


def test_durable_branch_has_exact_step_slots_and_official_callback(inputs,tmp_path):
    from experiments.spectral_reachability.pusht_full_packet_physical_20261027 import image_receipt
    plan=m.plans_from_saved(*inputs)[0];env=FakeEnv();calls=[]
    def evaluate(state):
        calls.append(state)
        return dict(official={'unchanged_callback':True},pose={'unchanged_callback':True})
    row=m.physical_branch(plan,tmp_path,lambda:{'bound':True},env,lambda e:np.array([e.n]*7),evaluate)
    assert row['status']=='completed' and len(calls)==1
    assert row['official']==row['pose']=={'unchanged_callback':True}
    assert m.sha(row['journal_path'])==row['journal_sha256']
    assert image_receipt(row,'prefix').name.endswith('_rgb_5.png')
    assert image_receipt(row,'terminal').name.endswith('_rgb_10.png')
    assert json.loads((tmp_path/'root_1_selected_imagined_replay0.physical.json').read_text())==row


def test_durable_reset_failure_retained(inputs,tmp_path):
    plan=m.plans_from_saved(*inputs)[0]
    def fail():raise RuntimeError('reset unavailable')
    row=m.physical_branch(plan,tmp_path,fail,None,None,None)
    assert row['status']=='branch_failed' and row['completed_commands']==0
    assert row['journal_sha256'] and row['failures']


def test_setup_failure_materializes_all_six_without_neural_work(inputs,tmp_path,monkeypatch):
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    plans=m.plans_from_saved(*inputs)
    p=dict(plans=plans,bindings={},planning_report='unused',feedback_dir='unused',encode_terminals=False)
    monkeypatch.setattr(m,'verify_prepared',lambda output:p)
    context=(base,None,None,None,None,None,None,None,None)
    monkeypatch.setattr(m,'prepare',lambda *args:(plans,{},context,[]))
    def fail(*args,**kwargs):raise RuntimeError('injected env setup failure')
    monkeypatch.setattr(base.gym,'make',fail)
    (tmp_path/'protocol.json').write_text('{}')
    (tmp_path/'prepare_receipt.json').write_text(json.dumps(dict(source_seals=[])))
    with pytest.raises(RuntimeError,match='integrity failed'):
        m.run(tmp_path)
    r=json.loads((tmp_path/'report.json').read_text())
    assert len(r['rows'])==6 and r['physical_env_step_attempts']==0 and r['new_predict_calls']==0
    assert r['P1_admission'] is False and all(z['status']=='branch_failed' for z in r['rows'])


def test_no_prediction_or_training_call_in_consumer():
    import inspect
    source=inspect.getsource(m)
    for forbidden in ('model.predict(', 'autograd.', 'optimizer.step(', 'bounded_ridge('):
        assert forbidden not in source
