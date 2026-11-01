"""NumPy/interface fakes and saved checks only, never native simulation/neural."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import p1_witness_physical_20261027_v1 as m


def rgb():
    image=np.full((224,224,3),255,np.uint8);image[30:40,30:40]=[65,105,225]
    return image


class Env:
    relative=True;action_scale=100;render_mode='rgb_array';render_size=224
    def __init__(self,fail_step=None,term=None,render_fail=None):
        self.unwrapped=self;self.state=np.array([100.,100.,200.,200.,0.,0.,0.]);self.goal_state=self.state.copy()
        self.block=type('Block',(),{'shapes':[]})();self.reset_calls=[];self.step_calls=[];self.renders=0
        self.fail_step=fail_step;self.term=term;self.render_fail=render_fail;self.closed=False
    def reset(self,seed,options=None):
        self.reset_calls.append((seed,deepcopy(options)))
        if options is not None:self.state=options['state'].copy();self.goal_state=options['goal_state'].copy()
        return {'state':self.state.copy()},{'goal_state':self.goal_state.copy(),'goal':rgb(),'goal_pose':np.array([200.,200.,0.])}
    def render(self):
        self.renders+=1
        if self.renders==self.render_fail:raise RuntimeError('render failure')
        return rgb()
    def _get_obs(self):return self.state.copy()
    def step(self,raw):
        self.step_calls.append(raw.copy());i=len(self.step_calls)
        if i==self.fail_step:raise RuntimeError('step primary')
        self.state[:2]+=raw
        return {'state':self.state.copy()},-1.,i==self.term,False,{'goal_state':self.goal_state.copy(),'n_contacts':0,'goal':rgb()}
    def eval_state(self,goal,state):return False,float(np.linalg.norm(goal-state))
    def close(self):self.closed=True


def fixture(monkeypatch,tmp_path):
    state=np.array([100.,100.,200.,200.,0.,0.,0.]);goal=state.copy()
    plan_source=dict(slots=[dict(candidate_id=0,root_seed=11,reset_seed_span=[100,104])],
                     reset_attempts_per_proposal=2,placement_reset_seed=0,rgb_shape=[224,224,3])
    metadata=dict(root_sha256=m.source.root_hash(state,goal),source=dict(proposal_index=0),
                  selected_attempt=dict(attempt_index=1,random_reset_seed=101,placement_reset_seed=0))
    packet=dict(state=state.copy(),goal=goal.copy(),rgb_history=np.stack([rgb()]*3),goal_rgb=rgb(),metadata=metadata)
    path=tmp_path/'raw.npz';path.write_bytes(b'fake_only')
    plan=dict(candidate_id=0,root_sha256=metadata['root_sha256'],arm='zero',selection_sha256='a'*64,
              raw_source_file=str(path),raw_source_sha256=m.sha(path),source_metadata=metadata,commands=np.zeros((10,2),np.float32).tolist())
    monkeypatch.setattr(m.source,'load_raw_source',lambda *args:deepcopy(packet))
    monkeypatch.setattr(m.source,'sample_state',lambda *args:state.copy())
    return plan,plan_source,packet


def run_branch(monkeypatch,tmp_path,env=None):
    plan,p,packet=fixture(monkeypatch,tmp_path);store=m.Store(tmp_path/'run')
    counts={k:dict(attempted=0,completed=0) for k in ('reset','render','step')};env=Env() if env is None else env
    row=m.branch(plan,p,env,store,counts)
    return row,counts,env,store,packet


def test_exact_original_reset_requests_all10_steps_three_png(monkeypatch,tmp_path):
    row,c,env,store,_=run_branch(monkeypatch,tmp_path)
    assert row['status']=='completed' and row['initial_exact'] and row['full_horizon'] and row['endpoint_rgb_fresh']
    assert c==dict(reset=dict(attempted=2,completed=2),render=dict(attempted=11,completed=11),step=dict(attempted=10,completed=10))
    assert env.reset_calls[0]==(101,None) and env.reset_calls[1][0]==0
    assert len(env.step_calls)==10 and all(x.dtype==np.float32 for x in env.step_calls)
    assert [x['step'] for x in row['checkpoints']]==[0,5,10]
    from PIL import Image
    for rec in row['checkpoints']:
        assert m.sha(rec['path'])==rec['file_sha256']
        assert np.array_equal(np.asarray(Image.open(rec['path'])),rgb())
    events=m.source.read_journal(store.path/'journal.jsonl')
    assert sum(e['kind']=='step_attempted' for e in events)==10
    assert sum(e['kind']=='step_returned' for e in events)==10
    assert sum(e['kind']=='fresh_rgb_observed' for e in events)==11
    assert row['guard_receipt']['primitive_commands_authorized']==10
    assert len(row['guard_receipt']['observations'])==11


def test_poststate_is_not_reset_input_and_mismatch_prevents_dispatch(monkeypatch,tmp_path):
    plan,p,packet=fixture(monkeypatch,tmp_path);packet['state'][0]=999
    monkeypatch.setattr(m.source,'load_raw_source',lambda *args:deepcopy(packet))
    env=Env();store=m.Store(tmp_path/'run');c={k:dict(attempted=0,completed=0) for k in ('reset','render','step')}
    row=m.branch(plan,p,env,store,c)
    assert row['status']=='failed' and not row['initial_exact'] and c['step']['attempted']==0
    assert env.reset_calls[1][1]['state'][0]==100 and env.reset_calls[1][1]['state'][0]!=packet['state'][0]


@pytest.mark.parametrize('stop',[1,4,5,10])
def test_termination_stops_no_replacement_no_stale_endpoint(monkeypatch,tmp_path,stop):
    row,c,env,_,_=run_branch(monkeypatch,tmp_path,Env(term=stop))
    assert row['status']=='environment_terminated' and c['step']['completed']==stop
    assert len(env.step_calls)==stop
    assert row['endpoint_rgb_fresh'] is (stop in (5,10))
    assert row['endpoint_scope']=='partial_or_unavailable'


def test_step_failure_preserves_attempt_return_gap(monkeypatch,tmp_path):
    row,c,env,store,_=run_branch(monkeypatch,tmp_path,Env(fail_step=3))
    assert row['status']=='failed' and row['first_error']['message']=='step primary'
    assert c['step']==dict(attempted=3,completed=2) and len(row['trace'])==3
    assert row['trace'][-1]['attempted'] and not row['trace'][-1]['returned']
    assert row['final_state'] and not row['endpoint_rgb_fresh']
    assert m.source.read_journal(store.path/'journal.jsonl')[-1]['kind']=='branch_finished'


def test_poststep_render_failure_preserves_return_and_attempt(monkeypatch,tmp_path):
    row,c,_,_,_=run_branch(monkeypatch,tmp_path,Env(render_fail=2))
    assert row['status']=='failed' and c['step']['completed']==1
    assert c['render']==dict(attempted=2,completed=1) and not row['endpoint_rgb_fresh']


def test_returned_arrays_saved_before_official_info_error(monkeypatch,tmp_path):
    class BadInfo(Env):
        def step(self,raw):
            obs,reward,t,tr,info=super().step(raw);info['unsupported']=object()
            return obs,reward,t,tr,info
    row,c,_,store,_=run_branch(monkeypatch,tmp_path,BadInfo())
    assert row['status']=='failed' and c['step']['completed']==1
    assert row['trace'][0]['observation']['state'] and row['trace'][0]['actual_state']
    events=m.source.read_journal(store.path/'journal.jsonl')
    assert any(e['kind']=='returned_observation' for e in events) and any(e['kind']=='actual_successor' for e in events)


@pytest.mark.parametrize('defect',['box','nan','shape','not_float32'])
def test_illegal_commands_rejected_without_clip(defect):
    a=np.zeros((10,2),np.float64)
    if defect=='box':a[0,0]=1.01
    if defect=='nan':a[0,0]=np.nan
    if defect=='shape':a=a[:5]
    if defect=='not_float32':a[0,0]=.1
    with pytest.raises(ValueError):m.raw_commands(a)


def test_checkpoint_guard_requires_224_and_visible_canonical_pixels():
    with pytest.raises(ValueError):m.RgbObservabilityGuard(root_id='mock',agent_rgb=(65,105,225)).observe(np.zeros((32,32,3),np.uint8),primitive_step=0)
    with pytest.raises(RuntimeError):m.RgbObservabilityGuard(root_id='mock',agent_rgb=(65,105,225)).observe(np.zeros((224,224,3),np.uint8),primitive_step=0)
    guard=m.RgbObservabilityGuard(root_id='mock',agent_rgb=(65,105,225));guard.observe(rgb(),primitive_step=0)
    assert guard.receipt()['observations'][0]['canonical_agent_pixels']==100


def test_disk_floor_before_directory_or_dispatch(monkeypatch,tmp_path):
    from collections import namedtuple
    Usage=namedtuple('Usage','total used free');monkeypatch.setattr(m.shutil,'disk_usage',lambda _:Usage(100,99,1))
    with pytest.raises(ValueError,match='reserve'):m.Store(tmp_path/'run')
    assert not (tmp_path/'run').exists()


def test_store_exclusive_global_quota_and_hashchain(tmp_path):
    store=m.Store(tmp_path/'run');store.emit('test',value=1)
    assert len(m.source.read_journal(store.path/'journal.jsonl'))==1
    with pytest.raises(FileExistsError):m.Store(tmp_path/'run')
    store.used=m.CAP
    with pytest.raises(ValueError,match='cap'):store.emit('oversize')


def test_import_does_not_load_simulator_or_neural():
    subprocess.run([sys.executable,'-c','import sys; from experiments.spectral_reachability import '+m.NAME+'; assert not any(x in sys.modules for x in ("torch","gymnasium","pymunk","stable_worldmodel"))'],check=True)


def test_contract_keeps_full_denominator_and_existing_fresh_rgb_guard():
    c=m.contract()
    assert c['branches']==8 and c['selected_steps']==80 and c['resets']==16 and c['explicit_renders']==88
    assert c['arms']==['zero','full_H2_GN','full_H2_gradient','two_stage']
    assert 'existing_RgbObservabilityGuard' in c['guard'] and not c['P1_admission'] and not c['equal_compute']


def test_existing_guard_abstains_before_next_command_when_pixels_missing(monkeypatch,tmp_path):
    class Missing(Env):
        def render(self):
            self.renders+=1
            return np.zeros((224,224,3),np.uint8) if self.renders==4 else rgb()
    row,c,env,store,_=run_branch(monkeypatch,tmp_path,Missing())
    assert row['status']=='failed' and c['step']==dict(attempted=3,completed=3)
    assert len(env.step_calls)==3 and row['guard_receipt']['abstained']
    assert row['guard_receipt']['observations'][-1]['canonical_agent_pixels']==0
    assert not row['endpoint_rgb_fresh']
    assert sum(e['kind']=='fresh_rgb_observed' for e in m.source.read_journal(store.path/'journal.jsonl'))==4


@pytest.mark.parametrize('turns',[-3,-2,-1,0,1,2,3])
@pytest.mark.parametrize('angle',[0.,.2,.4,np.pi,-np.pi])
def test_independent_pose_invariant_to_goal_plus_minus_two_pi(turns,angle):
    goal=np.zeros(7);state=np.zeros(7);state[4]=angle
    base=m.pose_diagnostics(goal,state)['independent']
    goal[4]+=turns*2*np.pi
    shifted=m.pose_diagnostics(goal,state)['independent']
    assert shifted['wrapped_angle_error']==pytest.approx(base['wrapped_angle_error'],abs=4e-15)
    assert shifted['object_position_error']==base['object_position_error']
    assert shifted['same_allxy_threshold_wrapped_success']==base['same_allxy_threshold_wrapped_success']


def test_native_negative_angle_false_positive_retained_not_fixed():
    goal=np.zeros(7);state=np.zeros(7);goal[4]=-4.;state[4]=3.
    d=m.pose_diagnostics(goal,state)
    assert d['native_formula']['angle_term']<0 and d['native_formula']['success']
    assert not d['independent']['same_allxy_threshold_wrapped_success']
    assert d['native_vs_wrapped_success_disagreement']
    shifted=goal.copy();shifted[4]+=2*np.pi
    assert not m.pose_diagnostics(shifted,state)['native_formula']['success']


def test_native_termination_honored_even_when_wrapped_disagrees(monkeypatch,tmp_path):
    class IncorrectTermination(Env):
        def step(self,raw):
            obs,reward,_,tr,info=super().step(raw)
            return obs,reward,True,tr,info
        def eval_state(self,goal,state):return False,1.
    row,c,_,_,_=run_branch(monkeypatch,tmp_path,IncorrectTermination())
    assert row['status']=='environment_terminated' and c['step']['completed']==1
    assert row['trace'][0]['termination_protocol_risk']['terminated_vs_eval_disagreement']


def test_evaluator_exception_cannot_erase_termination_info_or_returned_arrays(monkeypatch,tmp_path):
    class BadEvaluator(Env):
        def eval_state(self,goal,state):raise RuntimeError('eval failure')
    row,c,_,store,_=run_branch(monkeypatch,tmp_path,BadEvaluator(term=1))
    assert row['first_error']['message']=='eval failure' and c['step']['completed']==1
    assert row['trace'][0]['terminated'] and row['trace'][0]['actual_state'] and row['official_info']
    assert any(e['kind']=='termination_returned' for e in m.source.read_journal(store.path/'journal.jsonl'))


@pytest.mark.parametrize('close_failure',[False,True])
def test_run_all_eight_denominator_global_counts_and_close_errors(monkeypatch,tmp_path,close_failure):
    plan,p,_=fixture(monkeypatch,tmp_path)
    p['slots'].append(dict(p['slots'][0],candidate_id=1))
    p.update(source_closure={str(Path(__file__).resolve()):m.sha(__file__)},environment_sha256=m.sha(__file__),
             reset_semantics=dict(internal_physics_steps_per_reset=2,internal_renders_per_reset=1))
    plans=[]
    for cid in (0,1):
        for arm in m.ARMS:
            plans.append(dict(plan,candidate_id=cid,arm=arm,search_cost={'mock':True}))
    prepared=dict(inputs=dict(plans=plans,source_plan=p,inherited_cost={}))
    monkeypatch.setattr(m,'verify',lambda *args:prepared)
    envs=[]
    class Closing(Env):
        def close(self):
            super().close()
            if close_failure and self is envs[0]:raise RuntimeError('close failure')
    def factory():
        env=Closing();envs.append(env);return env
    result=m.run('mock','a'*64,tmp_path/'run',factory,authorize_physics=True)
    assert len(result['rows'])==8 and len(envs)==8 and all(e.closed for e in envs)
    assert result['counts']=={k:dict(attempted=n,completed=n) for k,n in dict(reset=16,render=88,step=80).items()}
    assert result['reset_internal_physics_steps']['completed_reset_implied']==32
    assert len(list((tmp_path/'run').glob('*.png')))==24
    assert result['status']==('partial_or_failed' if close_failure else 'physical_calibration_recorded')
    if close_failure:assert result['rows'][0]['secondary_errors'][0]['phase']=='close'


def test_native_initial_saved_calibration2_has_no_observed_false_success():
    # Saved NumPy arrays only, no environment import; skip outside this worktree.
    directory=m.ROOT/'results/p1_conditional_calibration2_main_20261027_v1'
    if not (directory/'sourcepack.json').is_file():pytest.skip('actual saved sourcepack unavailable')
    saved=m.read(directory/'sourcepack.json')
    assert m.sha(directory/'sourcepack.json')=='728f14cb83c4fcd0ea4f5348bad2bff5b5dd372ab0185b6e6b3a26dbfda58057'
    for row in saved['slots']:
        packet=m.source.load_raw_source(directory/row['raw_source_file'],saved['plan'])
        d=m.pose_diagnostics(packet['goal'],packet['state'])
        assert packet['goal'][4]<0
        assert not d['native_formula']['negative_angle_term'] and not d['native_formula']['success']
        assert not d['native_vs_wrapped_success_disagreement']


def test_actual_saved_selections_and_stable_root_join_under_top_array_reordering(monkeypatch):
    folder=m.ROOT/'results/p1_two_stage_witness_20261027_v1_cuda_main01'
    if not (folder/'report.json').is_file():pytest.skip('actual complete witness unavailable')
    original=m.read
    def reordered(path):
        value=original(path)
        if Path(path).resolve()==folder/'report.json':value['roots'].reverse()
        if Path(path).name=='report.json' and 'direct' in value:value['direct'].reverse()
        return value
    monkeypatch.setattr(m,'read',reordered)
    data=m.collect(
        m.ROOT/'results/p1_two_stage_witness_20261027_v1_prepared01','1ade88fd784f2a1217ede182d8e46f99b3007b902406c97413f46ed70cc16106',
        folder/'report.json','d85ad0999f316db9860bc5358f787488cadfd76c5bc8e2a0ff5a59439e5903c5',
        m.ROOT/'results/p1_direct_cuda_calibration_main_20261027_v2_prepared01','24eaf8b02873b3d93ebeeadbead9ca56c2caa079cea8c30b308357ddd8a3d2e6',
        m.ROOT/'results/p1_direct_cuda_calibration_main_20261027_v2_cuda_main01/report.json','fad2c2ff4f409762fa5925f142f9b4d3edc371645fc4c114bfa69c996483f3d4',
        m.ROOT/'results/p1_conditional_calibration2_main_20261027_v1','728f14cb83c4fcd0ea4f5348bad2bff5b5dd372ab0185b6e6b3a26dbfda58057')
    assert [(p['candidate_id'],p['arm']) for p in data['plans']]==[(cid,arm) for cid in (0,1) for arm in m.ARMS]
    assert len({p['root_sha256'] for p in data['plans']})==2
    for p in data['plans']:
        assert len(p['commands'])==10 and m.raw_commands(p['commands']).dtype==np.float32
        assert p['selection_sha256']==m.source.digest({k:v for k,v in p.items() if k!='selection_sha256'})


def test_prepare_rejects_missing_actual_report_before_upstream_verification(tmp_path):
    with pytest.raises(ValueError,match='binding drift'):
        m.collect(tmp_path,'a'*64,tmp_path/'missing.json','b'*64,tmp_path,'c'*64,tmp_path/'missingdirect.json','d'*64,tmp_path,'e'*64)
