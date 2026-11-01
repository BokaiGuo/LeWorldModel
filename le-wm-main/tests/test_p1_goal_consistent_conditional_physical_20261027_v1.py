"""Algebra/fake-env only; no simulator, neural inference or downloads."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import p1_goal_consistent_conditional_physical_20261027_v1 as m


def plan(root=0,arm='old_Q'):
    p=dict(schema=m.NAME+'.plan',root_index=root,root_sha256=str(root)*64,method=arm,
        planned_horizon_tokens=2,planned_commands=10,corrected_search_horizon=2,
        commands=np.zeros((10,2)).tolist(),goal=np.zeros(192).tolist(),
        projections={a:np.diag(([1.,0.] if a=='old_Q' else [0.,1.])+[0.]*190).tolist() for a in m.ARMS},
        active_metric_id='metric',export_selection_sha256='export',compiler_report_sha256='report')
    p['selection_sha256']=m.digest(p);return p


class Env:
    def __init__(self,missing=None,fail=None,renderfail=None,terminate=None):
        self.n=0;self.missing=missing;self.fail=fail;self.renderfail=renderfail;self.terminate=terminate
    def render(self):
        if self.n==self.renderfail:raise RuntimeError('fixture render')
        rgb=np.zeros((224,224,3),np.uint8)
        if self.n!=self.missing:rgb[20:22,20:22]=[65,105,225]
        return rgb
    def step(self,command):
        self.n+=1
        if self.n==self.fail:raise RuntimeError('fixture step')
        return None,0.,self.n==self.terminate,False,{}
    def reset(self,*a,**k):self.n=0;return None,{}


def branch(tmp_path,env=None,p=None,reset=lambda:{'context':'same'}):
    return m.physical_branch(plan=p or plan(),output=tmp_path,reset=reset,env=env or Env(),
        state_of=lambda e:np.full(7,e.n,float),
        evaluate=lambda state:dict(official={'official_state_distance':float(state[0])},pose={'pose_cost':float(state[0])}))


def test_all_ten_commands_and_fresh_endpoint(tmp_path):
    p=plan();row=branch(tmp_path,p=p);m.verify_branch(p,row)
    assert row['full_horizon_completed'] and row['endpoint_scope']=='full_H2'
    assert row['physical_env_step_attempts']==row['physical_env_step_returns']==10
    assert len(list(tmp_path.glob('*.png')))==11


@pytest.mark.parametrize('args,a,b,fresh',[
    ({'missing':0},0,0,True),({'missing':6},6,6,True),({'fail':7},7,6,False),
    ({'renderfail':7},7,7,False),({'terminate':7},7,7,True)])
def test_all_guard_failure_partials_remain(tmp_path,args,a,b,fresh):
    row=branch(tmp_path,Env(**args));m.verify_branch(plan(),row)
    assert (row['physical_env_step_attempts'],row['physical_env_step_returns'])==(a,b)
    assert row['endpoint_fresh'] is fresh and not row['full_horizon_completed']


@pytest.mark.parametrize('damage',['horizon','count','arm','bounds','nan','hash'])
def test_invalid_plan_before_reset_or_dispatch(tmp_path,damage):
    p=plan()
    if damage=='horizon':p['planned_horizon_tokens']=3
    elif damage=='count':p['planned_commands']=15
    elif damage=='arm':p['method']='uniform'
    elif damage=='bounds':p['commands'][0][0]=1.1
    elif damage=='nan':p['commands'][0][0]=float('nan')
    else:p['selection_sha256']='bad'
    with pytest.raises(ValueError):branch(tmp_path,p=p,reset=lambda:pytest.fail('no reset'))


def test_step_meter_separates_budgets_charges_failed_calls_and_branch_cap():
    env=Env(fail=2);events=[];meter=m.StepMeter(env,events.append)
    meter.bind(plan());meter.phase='selected'
    meter.step([0.,0.])
    with pytest.raises(RuntimeError):meter.step([0.,0.])
    assert meter.counts['selected']==dict(attempts=2,returns=1)
    meter.phase='warmup';meter.step([0.,0.])
    assert meter.counts['warmup']==dict(attempts=1,returns=1)
    meter.phase='selected'
    for _ in range(8):meter.step([0.,0.])
    with pytest.raises(ValueError,match='budget'):meter.step([0.,0.])
    assert env.n==11  # blocked eleventh selected never dispatched


def test_partial_pair_is_not_complete_pair():
    plans=[plan(i,a) for i in range(4) for a in m.ARMS];rows=[]
    for p in plans:
        rows.append(dict(root_sha256=p['root_sha256'],method=p['method'],selection_sha256=p['selection_sha256'],
            planned_horizon_tokens=2,planned_commands=10,context_identity={'same':True},endpoint_fresh=True,
            full_horizon_completed=p['root_index']!=1,physical_env_step_returns=6 if p['root_index']==1 else 10,
            pose={'pose_cost':2. if p['method']=='goal_patch_Q' else 3.},official={'official_state_distance':1.}))
    pairs=m.comparisons(plans,rows)
    assert [p['classification'] for p in pairs]==['full_H2_pair','partial_same_steps','full_H2_pair','full_H2_pair']
    assert pairs[1]['new_minus_old_pose_cost'] is None
    assert pairs[0]['new_minus_old_pose_cost']==-1.


def test_no_encoder_before_physical_report(tmp_path):
    with pytest.raises(ValueError):m.optional_encodings([],[],tmp_path,lambda event:pytest.fail('no CUDA'))


@pytest.mark.parametrize('args,attempts,status',[({},1,'completed'),({'missing':6},1,'completed'),
    ({'fail':7},0,'blocked_stale_endpoint'),({'renderfail':7},0,'blocked_stale_endpoint')])
def test_one_endpoint_latent_two_q_costs(tmp_path,args,attempts,status):
    p=plan();r=branch(tmp_path,Env(**args));m.write(tmp_path/'physical_report.json',{'rows':[r]})
    calls=[]
    def setup(event):
        def encode(path):calls.append(str(path));return np.array([2.,3.]+[0.]*190)
        return encode
    receipt=m.optional_encodings([p],[r],tmp_path,setup)[0]
    assert receipt['status']==status and receipt['attempted_encode_samples']==attempts
    assert len(calls)==attempts
    if attempts:
        assert receipt['actual_Q_costs']=={'old_Q':4/192,'goal_patch_Q':9/192}


def test_endpoint_module_encoder_projector_attempts_are_independent():
    events=[];stats=m.empty_encoding_stats();counter=m.EncodingCap(stats,events.append)
    for name,shape in [('encoder',(1,3,224,224)),('projector',(1,192))]:
        x=SimpleNamespace(device=SimpleNamespace(type='cuda'),shape=shape)
        for _ in range(8):counter.dispatch(name,lambda x:x,(x,),{})
        with pytest.raises(ValueError,match='cap'):counter.dispatch(name,lambda x:x,(x,),{})
        assert stats[name+'_attempts']==stats[name+'_completed']==8
    assert stats['forbidden_predict_attempts']==0


def test_import_does_not_load_torch_or_simulator():
    import subprocess,sys
    subprocess.run([sys.executable,'-c','import sys; from experiments.spectral_reachability import '+m.NAME+
        '; assert "torch" not in sys.modules; assert "gymnasium" not in sys.modules; assert "pymunk" not in sys.modules'],check=True)


def context():
    return dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
        latent_history=np.zeros((3,192)).tolist(),action_history=np.zeros((2,10)).tolist())


def test_context_tokens_rebuilt_in_time_major_normalized_space():
    initial=context();raw=(np.arange(20,dtype=np.float32).reshape(10,2)-10)/20
    after=deepcopy(initial);after['action_history']=m.tokens(raw).tolist()
    m.context_witness(initial,raw,after)
    assert not np.array_equal(after['action_history'],raw.reshape(2,10))
    prefix=deepcopy(initial);prefix['action_history']=[initial['action_history'][-1],m.tokens(raw[:5])[0].tolist()]
    m.context_witness(initial,raw[:5],prefix)


@pytest.mark.parametrize('damage',['layout','dim','history_value','history_order','history_raw','metadata'])
def test_context_layout_or_token_content_not_just_shape_is_validated(damage):
    initial=context();raw=np.arange(20,dtype=np.float32).reshape(10,2)/20
    after=deepcopy(initial);after['action_history']=m.tokens(raw).tolist()
    if damage=='layout':after['layout']='other'
    elif damage=='dim':after['action_dim']=True
    elif damage=='history_value':after['action_history'][0][0]+=1e-5
    elif damage=='history_order':after['action_history'].reverse()
    elif damage=='history_raw':after['action_history']=raw.reshape(2,10).tolist()
    else:after['extra']='unbound'
    with pytest.raises(ValueError):m.context_witness(initial,raw,after)


def test_encoder_failed_call_charged_without_projector_refund():
    stats=m.empty_encoding_stats();cap=m.EncodingCap(stats,lambda e:None)
    x=SimpleNamespace(device=SimpleNamespace(type='cuda'),shape=(1,3,224,224))
    def fail(x):raise RuntimeError('fixture encoder')
    with pytest.raises(RuntimeError):cap.dispatch('encoder',fail,(x,),{})
    assert stats['encoder_attempts']==1 and stats['encoder_completed']==0
    assert stats['projector_attempts']==0


def test_cleanup_failure_does_not_replace_first_error():
    class Owner:
        def __setattr__(self,key,value):raise RuntimeError('fixture restore')
    failures=[{'phase':'encode','type':'FirstError','message':'preserve first'}]
    m.restore_hooks([(Owner(),'forward',None)],failures)
    assert failures[0]['type']=='FirstError'
    assert failures[1]['phase']=='hook_restore' and failures[1]['attribute']=='forward'


@pytest.fixture(scope='module')
def actual_arrays():
    # Read-only actual main-provided arrays, not a new neural/physical run.
    out=m.PROJECT/'results/p3_goal_patch_consistent_compiler_20261027_v1_cuda_main_v1'
    p=m.original_inputs(m.PROJECT)
    raw=(out/'report.json').read_bytes();p['compiler_report_sha256']=m.io.sha(raw)
    return p,out,raw,m.planning.selected_actions(out)


def test_actual_all80_histories_and_eight_plans_pass_independent_gate(actual_arrays):
    p,out,raw,exports=actual_arrays
    a=m.verify_compiler_histories(p,out,exports)
    assert a['candidates']==80 and a['replay_contexts']==104 and a['selected_exports']==8 and a['all_checks']
    plans=m.plans_from_exports(p,exports)
    assert len(plans)==8 and sum(p['planned_commands'] for p in plans)==80
    assert [p['method'] for p in plans]==[a for group in p['arm_order'] for a in group]


@pytest.mark.parametrize('damage',['layout','history'])
def test_nonselected_candidate_bad_content_is_rejected_not_only_winner(monkeypatch,actual_arrays,damage):
    p,out,raw,exports=actual_arrays;decode=m.io.decode;report=decode(raw)
    compilation=report['rows'][0]['compilation'];index=(compilation['winner']['index']+1)%10
    bad=compilation['candidates'][index]['context']
    if damage=='layout':bad['layout']='invalid'
    else:bad['action_history'][0][0]+=1e-6
    monkeypatch.setattr(m.io,'decode',lambda value:report if value==raw else decode(value))
    with pytest.raises(ValueError):m.verify_compiler_histories(p,out,exports)


@pytest.mark.parametrize('damage',['root','horizon','Q','goal','hash','actions'])
def test_export_mismatch_not_dispatched(actual_arrays,damage):
    p,out,raw,exports=actual_arrays;e=deepcopy(exports[0]);others=exports[1:]
    if damage=='root':e['root_sha256']='bad'
    elif damage=='horizon':e['horizon']=3
    elif damage=='Q':e['projection'][0][0]+=.1
    elif damage=='goal':e['goal'][0]+=.1
    elif damage=='actions':e['actions'][0][0]=1.5
    else:e['selection_sha256']='bad'
    with pytest.raises(ValueError):m.plans_from_exports(p,[e,*others])


def test_full_fake_host_80steps_separate120warmup_no_model_claim(monkeypatch,tmp_path,actual_arrays):
    import sys
    from experiments import spectral_reachability as package
    p,out,raw,exports=actual_arrays
    protocol=deepcopy(p);protocol.update(plans=m.plans_from_exports(p,exports),stage='report_bound_ready')
    class HostEnv(Env):
        def __init__(self):
            super().__init__();self.unwrapped=self
            self.variation_space={'agent':{'color':SimpleNamespace(value=np.array([65,105,225]))}}
        def eval_state(self,goal,state):return False,0.
        def close(self):raise RuntimeError('fixture close failure retained')
    env=HostEnv()
    evaluator=Path(HostEnv.eval_state.__code__.co_filename).resolve()
    protocol['sources_sha256']={str(evaluator):m.io.sha(evaluator.read_bytes())}
    roots=[dict(state=np.array([i]+[0.]*6),goal=np.zeros(7)) for i in range(4)]
    def collect(meter,**kwargs):
        for _ in range(32):meter.reset()
        return roots
    def warmup(meter,root,history_len):
        meter.reset()
        for _ in range(15):meter.step([0.,0.])
        return [env.render()]*3,np.zeros((3,2)),env.render()
    base=SimpleNamespace(gym=SimpleNamespace(make=lambda *a,**k:env),
        collect_nonterminal_boundary_contact_roots=collect,
        root_fingerprint=lambda root:protocol['roots'][int(root['state'][0])],
        physical=SimpleNamespace(_reset_and_warmup=warmup,_pose_label=lambda s,g:{'pose_cost':float(s[0])}),
        p0=SimpleNamespace(packets=SimpleNamespace(state_of=lambda e:np.full(7,e.n,float))))
    monkeypatch.setitem(sys.modules,'experiments.spectral_reachability.audit_pusht_d2p_goal_agent_intervention',base)
    monkeypatch.setattr(package,'audit_pusht_d2p_goal_agent_intervention',base,raising=False)
    sampling=SimpleNamespace(verify_rgb_context=lambda f,g,r:deepcopy(r),
        official_checked=lambda e,s,g:{'official_state_distance':float(s[0])})
    monkeypatch.setitem(sys.modules,'experiments.spectral_reachability.run_pusht_sampling_comparison_20261027_v2',sampling)
    monkeypatch.setattr(m,'validate_prepared',lambda *a:protocol)
    output=tmp_path/'physical'
    with pytest.raises(RuntimeError,match='preserve all physical'):m.execute_physics(tmp_path/'prep','sha',output)
    r=m.io.decode((output/'report.json').read_bytes())
    assert r['physical_env_step_attempts']==r['physical_env_step_returns']==80
    assert r['physical_resources']['steps']['warmup']==dict(attempts=120,returns=120)
    assert r['physical_resources']['resets']['root_collection']==dict(attempts=32,returns=32)
    assert len(r['rows'])==8 and all(row['full_horizon_completed'] for row in r['rows'])
    assert r['model_loaded'] is False and r['model_unchanged'] is None and r['model_hash_verified'] is False
    assert r['failures'][0]['phase']=='env_close'
