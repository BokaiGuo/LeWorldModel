"""Fake-only producer tests: NEVER construct native physics or CUDA."""
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import p1_observed_history_source_20261027_v1 as m


class Shape:
    def get_vertices(self):
        from types import SimpleNamespace
        return [SimpleNamespace(x=x,y=y) for x,y in [(-10,-10),(10,-10),(10,10),(-10,10)]]
    def point_query(self,point):
        from types import SimpleNamespace
        return SimpleNamespace(distance=15.)


class Fake:
    def __init__(self,terminal_at=None,fail_step=None,close_fail=False):
        from types import SimpleNamespace
        self.unwrapped=self;self.block=SimpleNamespace(shapes=[Shape()],velocity=(0.,0.),angular_velocity=0.)
        self.agent=SimpleNamespace(shapes=[SimpleNamespace(radius=15.)])
        self.state=np.zeros(7);self.calls=[];self.n=0;self.terminal_at=terminal_at
        self.fail_step=fail_step;self.close_fail=close_fail;self.dt=.01;self.control_hz=10
    def _get_obs(self):return self.state.copy()
    def _set_state(self,value):
        self.state=np.asarray(value).copy();self.space.step(self.dt);self.state[0]+=.125
    def render(self):return np.full((4,4,3),round(self.state[0])%256,np.uint8)
    def reset(self,seed,options=None):
        from types import SimpleNamespace
        self.calls.append((seed,deepcopy(options)));self.n=0
        self.space=SimpleNamespace(step=lambda dt:None)
        self.goal_state=np.array([200.,200.,250.,250.,0.,0.,0.]) if options is None else options['goal_state'].copy()
        self._set_state(self.goal_state);self.goal_rgb=self.render()
        self._set_state(self.goal_state if options is None else options['state'])
        return {'state':self.state.copy()},{'goal_state':self.goal_state.copy(),'goal':self.goal_rgb.copy()}
    def eval_state(self,goal,state):return False,1.
    def step(self,action):
        assert np.array_equal(action,np.zeros(2,np.float32));self.n+=1
        if self.fail_step==self.n:raise RuntimeError('step primary')
        for _ in range(10):self.space.step(self.dt)
        self.state[0]+=.5
        return {'state':self.state.copy()},0.,self.n==self.terminal_at,False,{'goal_state':self.goal_state.copy(),'goal':self.goal_rgb.copy()}
    def close(self):
        if self.close_fail:raise RuntimeError('close secondary')


def fixture(monkeypatch):
    plan=dict(slots=[dict(candidate_id=i,root_seed=100+i,reset_seed_span=[1000+4*i,1004+4*i],split='calibration') for i in range(2)],
        placement_reset_seed=0,reset_attempts_per_proposal=2,rgb_shape=[4,4,3],
        recipe=dict(position_jitter=16.,angle_offset_range=[-.8,.8],edge_fraction_range=[.2,.8],pusher_surface_offset=15.),
        domain=dict(canvas_xy=[0.,512.],contact_gap_range=[-1.,1.],nonterminal_position_tolerance=1.,nonterminal_angle_tolerance=.01),
        environment_sha256=m.source.file_sha(__file__),source_closure={str(Path(__file__).resolve()):m.source.file_sha(__file__)})
    packets={};rows=[];operations={}
    for slot in plan['slots']:
        env=Fake();random=env.reset(slot['reset_seed_span'][0]);goal=random[1]['goal_state']
        requested=m.source.sample_state(goal,env.block.shapes,slot,0,0,plan)
        placed=env.reset(0,dict(state=requested,goal_state=goal));frame=env.render()
        cid=slot['candidate_id'];root=m.source.root_hash(placed[0]['state'],goal)
        packets[cid]=dict(state=placed[0]['state'],goal=goal,goal_rgb=placed[1]['goal'],rgb_history=np.stack([frame]*3),
            metadata=dict(source=dict(proposal_index=0),selected_attempt=dict(attempt_index=0,random_reset_seed=slot['reset_seed_span'][0],placement_reset_seed=0),root_sha256=root))
        rows.append(dict(slot=slot,metadata=packets[cid]['metadata']))
        operations[cid,0,0,'random_reset']=m.source.reset_summary(random,plan)
        operations[cid,0,0,'placement_reset']=m.source.reset_summary(placed,plan)
        operations[cid,0,0,'render']=dict(rgb_sha256=m.image_sha(frame))
    data=(dict(plan=plan),dict(slots=rows),operations,packets)
    prepared=dict(schema=m.NAME+'.prepare',normalization=dict(mean=[.1,-.2],std=[.5,2.],training_provenance_verified=False),bindings={},
        rgb_shape=[4,4,3],selections=[dict(candidate_id=r['slot']['candidate_id'],source=r['metadata']['source'],
            selected_attempt=r['metadata']['selected_attempt'],parent_root_sha256=r['metadata']['root_sha256']) for r in rows])
    monkeypatch.setattr(m,'verify',lambda p:data)
    return prepared,data


def test_complete_observed_history_and_exact_budgets(monkeypatch,tmp_path):
    p,data=fixture(monkeypatch);env=Fake()
    r=m.execute(p,tmp_path/'run',lambda:env,authorization=m.AUTH)
    assert r['status']=='observed_calibration_source_completed'
    assert {k:v['completed'] for k,v in r['counts'].items()}==dict(reset=4,public_step=20,explicit_render=6,set_state=8,goal_render=4,physics_step=208)
    assert r['durable_journal_complete'] and r['first_error'] is None
    for row in r['rows']:
        packet=m.load_observed(tmp_path/'run'/row['file'],p)
        assert packet['rgb_history'].shape==(3,4,4,3)
        assert not np.array_equal(packet['rgb_history'][0],packet['rgb_history'][2])
        assert np.array_equal(packet['state'],packet['observed_states'][-1])
        assert packet['metadata']['root_sha256']==m.source.root_hash(packet['state'],packet['goal'])
        assert np.array_equal(packet['normalized_action_history'],np.tile(np.array([-.2,.1],np.float32),(2,5)))
        assert packet['metadata']['actual_observed_frames']==3
    # Native setter drift .125 means feeding saved poststate would compound it.
    expected=m.source.sample_state(data[3][0]['goal'],[Shape()],data[0]['plan']['slots'][0],0,0,data[0]['plan'])
    assert np.array_equal(env.calls[1][1]['state'],expected)
    assert not np.array_equal(env.calls[1][1]['state'],data[3][0]['state'])


def test_terminal_kept_failed_without_replacement(monkeypatch,tmp_path):
    p,_=fixture(monkeypatch);r=m.execute(p,tmp_path/'run',lambda:Fake(terminal_at=3),authorization=m.AUTH)
    assert r['status']=='failed' and len(r['rows'])==2
    assert all(row['status']=='failed' for row in r['rows'])
    assert r['counts']['reset']['completed']==4 and r['counts']['public_step']['completed']==6
    assert not list((tmp_path/'run').glob('source_*.npz'))
    assert len(list((tmp_path/'run').glob('partial_*.npz')))==2
    with np.load(tmp_path/'run'/'partial_0.npz',allow_pickle=False) as z:
        assert z['raw_action_history'].shape==(3,2)
        assert z['normalized_action_history'].shape==(0,10)
    with pytest.raises(ValueError,match='complete observed'):
        m.load_observed(tmp_path/'run'/'partial_0.npz',p)


def test_first_error_survives_close(monkeypatch,tmp_path):
    p,_=fixture(monkeypatch);r=m.execute(p,tmp_path/'run',lambda:Fake(fail_step=2,close_fail=True),authorization=m.AUTH)
    assert r['first_error']['message']=='step primary'
    assert any(e['phase']=='close' for e in r['secondary_errors'])
    assert r['counts']['public_step']==dict(attempted=2,completed=1)
    assert r['rows'][1]['status']=='not_started'


def test_bad_auth_storage_and_existing_output_before_factory(monkeypatch,tmp_path):
    p,_=fixture(monkeypatch)
    with pytest.raises(ValueError):m.execute(p,tmp_path/'run',lambda:pytest.fail('factory'),authorization='NO')
    (tmp_path/'run').mkdir()
    with pytest.raises(FileExistsError):m.execute(p,tmp_path/'run',lambda:pytest.fail('factory'),authorization=m.AUTH)
    from collections import namedtuple
    U=namedtuple('U','total used free');monkeypatch.setattr(m.shutil,'disk_usage',lambda _:U(100,99,1))
    with pytest.raises(ValueError,match='reserve'):m.execute(p,tmp_path/'other',lambda:pytest.fail('factory'),authorization=m.AUTH)


def test_import_never_imports_environment_or_torch():
    subprocess.run([sys.executable,'-c','import sys; from experiments.spectral_reachability import '+m.NAME+'; assert not any(x in sys.modules for x in ("torch","gymnasium","stable_worldmodel","pymunk"))'],check=True)


def test_final_invalid_root_rejected_without_replacement(monkeypatch,tmp_path):
    p,_=fixture(monkeypatch)
    class Invalid(Fake):
        def step(self,action):
            value=super().step(action)
            if self.n==10:self.state[0]=900;value[0]['state']=self.state.copy()
            return value
    r=m.execute(p,tmp_path/'run',Invalid,authorization=m.AUTH)
    assert r['status']=='failed' and r['counts']['public_step']['completed']==20
    assert all(row['status']=='failed' and 'outside canvas' in row['failure_reason'] for row in r['rows'])
    assert len(list((tmp_path/'run').glob('partial_*.npz')))==2


def test_native_terminal_at_t0_does_not_step(monkeypatch,tmp_path):
    p,_=fixture(monkeypatch)
    class AlreadyTerminal(Fake):
        def eval_state(self,g,s):return True,0.
    r=m.execute(p,tmp_path/'run',AlreadyTerminal,authorization=m.AUTH)
    assert r['status']=='failed' and r['counts']['public_step']['attempted']==0
    assert r['counts']['reset']['completed']==4


def test_partial_failed_physics_count_retained(monkeypatch,tmp_path):
    p,_=fixture(monkeypatch)
    class Partial(Fake):
        def step(self,action):
            self.space.step(self.dt);raise RuntimeError('after one internal step')
    r=m.execute(p,tmp_path/'run',Partial,authorization=m.AUTH)
    assert r['counts']['reset']['completed']==2
    assert r['counts']['public_step']==dict(attempted=1,completed=0)
    assert r['counts']['physics_step']['completed']==5
    assert r['internal_physics']['completed_public_steps_imply']==0


def test_journal_fsynced_before_each_external_dispatch(monkeypatch,tmp_path):
    p,_=fixture(monkeypatch)
    class Checking(Fake):
        def step(self,action):
            events=m.source.read_journal(tmp_path/'run'/'journal.jsonl')
            assert events[-1]['kind']=='operation_started' and events[-1]['resource']=='public_step'
            assert list((tmp_path/'run').glob('checkpoint_*_0.npz'))
            if self.n>=5:assert list((tmp_path/'run').glob('checkpoint_*_5.npz'))
            return super().step(action)
    r=m.execute(p,tmp_path/'run',Checking,authorization=m.AUTH)
    events=m.source.read_journal(tmp_path/'run'/'journal.jsonl')
    assert events[-1]['kind']=='run_finished' and events[-1]['counts']==r['counts']
    assert sum(e['kind']=='operation_started' and e.get('resource')=='physics_step' for e in events)==208


def test_source_hash_failure_is_secondary_to_step_error(monkeypatch,tmp_path):
    p,_=fixture(monkeypatch);p['bindings']={str(Path(__file__).resolve()):'0'*64}
    r=m.execute(p,tmp_path/'run',lambda:Fake(fail_step=2),authorization=m.AUTH)
    assert r['first_error']['message']=='step primary'
    assert any(e['phase']=='final_binding' for e in r['secondary_errors'])


@pytest.mark.parametrize('defect',['normalized','root','indices','raw','schema','parent','normalizer','nonfinite'])
def test_observed_loader_rejects_corruption(monkeypatch,tmp_path,defect):
    p,_=fixture(monkeypatch);m.execute(p,tmp_path/'run',Fake,authorization=m.AUTH)
    path=tmp_path/'run'/'source_0.npz'
    with np.load(path,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    if defect=='normalized':arrays['normalized_action_history'][0,0]=0
    elif defect=='root':arrays['state'][0]+=1
    elif defect=='indices':arrays['observed_public_step_indices'][1]=4
    elif defect=='raw':arrays['raw_action_history'][0,0]=.1
    elif defect=='nonfinite':arrays['observed_states'][0,0]=np.nan
    else:
        meta=json.loads(str(arrays['metadata_json']));meta['schema']='legacy_padding';arrays['metadata_json']=np.asarray(json.dumps(meta))
        if defect in ('parent','normalizer'):
            meta['schema']=m.NAME+'.raw_observed_source'
            if defect=='parent':meta['parent_root_sha256']='0'*64
            else:meta['normalization']['mean'][0]=999
            arrays['metadata_json']=np.asarray(json.dumps(meta))
    with (tmp_path/'corrupt.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    with pytest.raises(ValueError):m.load_observed(tmp_path/'corrupt.npz',p)


def test_real_input_prepare_is_metadata_only_and_verifies(tmp_path):
    root=m.source.PROJECT/'results'
    request=root/'p1_conditional_calibration2_main_20261027_v1_request.json'
    report=root/'p1_conditional_calibration2_main_20261027_v1/sourcepack.json'
    audit=root/'p1_source_reset_goal_audit_20261027_v1_main01/report.json'
    value=m.prepare(tmp_path/'prepare.json',request=request,report=report,request_sha256=m.source.file_sha(request),
        report_sha256=m.source.file_sha(report),reset_audit_report=audit,reset_audit_sha256=m.source.file_sha(audit))
    assert value['counts']==m.COUNTS and len(value['selections'])==2
    assert m.verify(value)[0]['plan']['purpose']=='calibration_exploration'
    value['raw_commands'][0][0]=.1
    with pytest.raises(ValueError,match='protocol drift'):m.verify(value)
