"""RAM-only artificial dynamics/rendering. No actual inputs, neural or physics."""
from copy import deepcopy
from io import BytesIO
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from PIL import Image
import pytest
from experiments.spectral_reachability import p1_moving_target_overlay_source_20261027_v1 as m

spec=importlib.util.spec_from_file_location('moving_overlay_ram_legacy_fixture',Path(__file__).with_name('test_p1_constructive_moving_history_source_20261027_v1.py'))
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
REQUEST=np.asarray([100.,100.,150.,150.,0.,0.,0.])


def png(a):
    b=BytesIO();Image.fromarray(a).save(b,format='PNG');return b.getvalue()


class RAMStore(h.MemoryStore):
    def __init__(self,*,fault=None):super().__init__();self.fault=fault
    def emit(self,kind,**fields):
        if kind==self.fault:raise OSError('mock fsync journal failure '+kind)
        return super().emit(kind,**fields)
    def frame(self,name,a):
        assert name not in self.files
        self.files[name]=png(a);return m.moving.image_sha(np.frombuffer(self.files[name],np.uint8))


class Env(h.FakeEnv):
    def __init__(self,*,fault=None,at=1,close_error=False):
        super().__init__(close_error=close_error);self.fault=fault;self.at=at;self.closed=False
        self.render_action=False;self.render_mode='rgb_array';self.render_size=224;self.window_size=512
        self.goal_pose=np.asarray([300.,300.,0.]);self.render_flags=[];self.on_seen=False
        self.variation_space=NS(value=dict(rendering=dict(render_goal=1),goal=dict(position=self.goal_pose[:2].copy(),
            color=np.asarray([0,255,0],np.uint8)),background=dict(color=np.asarray([255]*3,np.uint8))))
    def eval_state(self,goal,state):
        success=np.array_equal(goal,state)
        if self.fault=='early' and len(self.commands)>=self.at:success=True
        return bool(success),float(np.linalg.norm(goal-state))
    def render(self):
        a=super().render();self.render_flags.append(self.with_target)
        if self.with_target:
            a[30:50,30:50]=[0,255,0];self.on_seen=True
            if self.fault=='render_state':self.state[0]+=1;self.agent.position=tuple(self.state[:2])
            if self.fault=='goal_pose':self.goal_pose[0]+=1
            if self.fault=='color':self.variation_space.value['goal']['color'][0]=1
            if self.fault=='goal':self.goal_state[0]+=1
            if self.fault=='render_raise':raise RuntimeError('primary render failure')
        elif self.fault=='off_replay' and self.on_seen:a[1,1]=[0,0,0]
        return a
    def step(self,c):
        v=super().step(c)
        if len(self.commands)==self.at:
            if self.fault=='substitute':self.latest_action=np.zeros_like(c)
            if self.fault=='dtype':self.latest_action=self.latest_action.astype(float)
            if self.fault=='step_raise':raise RuntimeError('primary step failure')
            if self.fault=='truncate':return v[0],v[1],v[2],True,v[4]
        return v
    def close(self):self.closed=True;return super().close()


def reference_fixture():
    slots=[dict(index=i,reset_seed=21+i,root_seed=11+i,sign=1 if i==0 else -1) for i in range(2)]
    data=dict(source_plan=dict(slots=slots,packet_logit_cap=.75*np.sqrt(5)),refs=[],witnesses=[],references={},bindings={})
    images={}
    for s,slot in enumerate(slots):
        construction=None
        for phase in ('construction','replay0','replay1'):
            env=Env();_,info=env.reset(seed=slot['reset_seed']);native=info['goal_state'].copy()
            obs,_=env.reset(seed=0,options=dict(state=REQUEST.copy(),goal_state=native))
            frames=[];states=[];dynamics=[];refmap={};key=f's{s}_{phase}'
            def capture(k):
                a=env.render();path=f'/RAM/{key}_{k}.png';b=png(a);images[path]=a.copy()
                refmap[str(k)]=dict(path=path,file_sha256=m.moving.image_sha(np.frombuffer(b,np.uint8)),
                    pixel_sha256=m.moving.image_sha(a),state=obs['state'].tolist(),snapshot=m.moving.snapshot(env))
                return a
            a=capture(0);frames.append(a);states.append(obs['state'].copy());dynamics.append(m.moving.snapshot(env))
            for k,c in enumerate(m.moving.warmup_commands(),1):
                obs,*_=env.step(c.copy());a=capture(k)
                if k in (5,10):frames.append(a);states.append(obs['state'].copy());dynamics.append(m.moving.snapshot(env))
            root=obs['state'].copy();_,commands=m.moving.witness(root,slot['sign'],data['source_plan']['packet_logit_cap'])
            if construction:m.moving.attach_goal(env,construction['goal'],construction['goal_rgb'])
            for k,c in enumerate(commands,11):obs,*_=env.step(c.copy());a=capture(k)
            if construction is None:
                construction=dict(root=root,goal=obs['state'].copy(),goal_rgb=a.copy(),history=np.asarray(frames),
                    states=np.asarray(states),dynamics=dynamics,slot=slot,root_sha256=m.moving.source.root_hash(root,obs['state']))
                data['refs'].append(construction);data['witnesses'].append(commands.copy())
            data['references'][key]=refmap
    return data,images


def run(monkeypatch,*,factory=Env,store=None,change=None):
    data,images=reference_fixture()
    if change:change(data,images)
    monkeypatch.setattr(m,'verify',lambda p:deepcopy(data))
    monkeypatch.setattr(m.moving,'validate_env',lambda body:None)
    monkeypatch.setattr(m.moving.source,'sample_state',lambda *a:REQUEST.copy())
    monkeypatch.setattr(m,'old_image',lambda r:images[r['path']].copy())
    envs=[]
    def make():e=factory();envs.append(e);return e
    store=RAMStore() if store is None else store
    r=m.execute(dict(schema=m.NAME+'.prepare'),store,make,authorization=m.AUTH)
    return r,store,envs,data


def test_complete_six_episodes_exact186render_census_real_on_terminal_and_blinds(monkeypatch):
    r,s,envs,data=run(monkeypatch)
    assert r['status']=='paired_moving_overlay_recorded',r['errors']
    assert r['full_protocol'] and r['errors']==[] and len(envs)==len(r['rows'])==6
    assert r['counts']=={k:dict(attempted=n,completed=n) for k,n in m.contract()['caps'].items()}
    assert r['new_independent_roots']==0 and r['P1_admission'] is False
    assert len([name for name in s.files if name.endswith('.png')])==186 and len(r['blind_exports'])==2
    fresh=[e for e in s.events if e['kind']=='fresh_overlay_frame']
    assert len(fresh)==186 and all(e['receipt']['reference_error'] is None for e in fresh)
    for row,env in zip(r['rows'],envs,strict=True):
        assert row['status']=='source_episode_completed' and row['public_returned']==20 and len(row['frames'])==31 and row['missing_triplets']==[]
        assert row['native_success']==(row['phase']!='construction') and env.closed and env.with_target is False
        assert all(k not in env.__dict__ for k in ('reset','render','_set_state','_set_goal_state'))
        assert np.array_equal(env.goal_pose,[300.,300.,0.])
        assert m.moving.exact(np.asarray(env.commands[:10]),m.moving.warmup_commands())
        assert m.moving.exact(np.asarray(env.commands[10:]),data['witnesses'][row['source_slot']])
        for step,start in [(0,0),(5,7),(10,14),(15,21),(20,28)]:
            fs=row['frames'][start:start+3]
            assert [(f['reference_key'],f['overlay_index']) for f in fs]==[(str(step),j) for j in range(3)]
            assert s.files[fs[0]['png']]==s.files[fs[2]['png']] and s.files[fs[0]['png']]!=s.files[fs[1]['png']]
    for e in fresh:
        f=e['receipt'];assert f['pre']==f['post']
        if not f['with_target']:assert all(f['old_off_match'].values())
        if f['overlay_index']==2:assert f['off_replay_match'] is True
    for entry in r['blind_exports']:
        arrays={}
        for flag,a in entry['artifacts'].items():
            assert m.moving.image_sha(np.frombuffer(s.files[a['path']],np.uint8))==a['sha256']
            with np.load(BytesIO(s.files[a['path']]),allow_pickle=False) as z:
                assert set(z.files)=={'rgb_history','raw_action_history','goal_rgb'}
                arrays[flag]={k:z[k].copy() for k in z.files}
                assert z['rgb_history'].shape==(3,224,224,3) and z['rgb_history'].dtype==np.uint8
                assert m.moving.exact(z['raw_action_history'],m.moving.warmup_commands())
                # Goal must be NEW construction true terminal render, not cache or synthetic.
                ref=[x for x in fresh if x['key']==f's{entry["source_slot"]}_construction' and x['receipt']['reference_key']=='20' and x['receipt']['overlay_index']==(0 if flag=='off' else 1)][0]['receipt']
                with Image.open(BytesIO(s.files[ref['png']])) as im:assert np.array_equal(z['goal_rgb'],np.asarray(im))
        assert not np.array_equal(arrays['off']['goal_rgb'],arrays['on']['goal_rgb'])
    # Full artifact+JSON journal+final report, not just compressed NPY arrays.
    total=sum(map(len,s.files.values()))+sum(len(m.raw(e)) for e in s.events)
    assert total<m.CAP and len(s.files['report.json'])<m.FINAL
    previous=None
    for e in s.events:
        assert e['previous_sha256']==previous and e['event_sha256']==m.digest({k:v for k,v in e.items() if k!='event_sha256'})
        previous=e['event_sha256']
    assert previous==r['journal_chain_tail']


@pytest.mark.parametrize('fault',['render_state','goal_pose','color','goal','off_replay'])
def test_render_mutation_rejected_after_new_png_receipt_and_flag_restored(monkeypatch,fault):
    r,s,envs,_=run(monkeypatch,factory=lambda:Env(fault=fault))
    assert r['status']=='engineering_failed' and not r['full_protocol'] and r['blind_exports']==[]
    assert all(e.closed and e.with_target is False for e in envs)
    assert len(r['rows'][0]['frames'])==(3 if fault=='off_replay' else 2)
    assert all(f['png'] in s.files for row in r['rows'] for f in row['frames'])
    assert any(e['kind']=='fresh_overlay_frame' and e['receipt']['with_target'] for e in s.events)


@pytest.mark.parametrize('field',['file','pixels','state','snapshot'])
def test_old_off_exactness_cannot_be_replaced_by_tolerance(monkeypatch,field):
    def change(data,images):
        for refs in data['references'].values():
            r=refs['0']
            if field=='file':r['file_sha256']='0'*64
            elif field=='pixels':images[r['path']][1,1]=255
            elif field=='state':r['state'][0]=np.nextafter(r['state'][0],999.)
            elif field=='snapshot':r['snapshot']['bodies']['block']['float64_bits']='wrong'
    r,s,envs,_=run(monkeypatch,change=change)
    assert r['status']=='engineering_failed' and len(r['rows'][0]['frames'])==1
    event=next(e for e in s.events if e['kind']=='fresh_overlay_frame')
    assert event['receipt']['old_off_match'][field] is False and event['receipt']['png'] in s.files


@pytest.mark.parametrize('fault',['early','truncate'])
def test_legal_warmup_stop_retains_six_rows_no_padding_no_engineering_census(monkeypatch,fault):
    r,s,envs,_=run(monkeypatch,factory=lambda:Env(fault=fault))
    assert r['status']=='partial_native_stop' and r['errors']==[] and len(r['rows'])==6
    assert [row['status'] for row in r['rows']]==['native_stopped','not_started','not_started']*2
    assert r['blind_exports']==[] and len(envs)==2 and all(len(e.commands)==1 for e in envs)
    assert r['counts']['public_step']==dict(attempted=2,completed=2)
    assert r['counts']['physics_step']==dict(attempted=28,completed=28)
    assert r['rows'][0]['missing_triplets']==['5','10','15','20']


@pytest.mark.parametrize('fault',['substitute','dtype','step_raise','render_raise'])
def test_actual_action_or_render_failures_keep_first_error_and_partial_census(monkeypatch,fault):
    r,s,envs,_=run(monkeypatch,factory=lambda:Env(fault=fault,close_error=True))
    assert r['status']=='engineering_failed' and r['first_error']['phase']=='source_episode'
    assert any(e['phase']=='close' for e in r['errors']) and all(e.with_target is False for e in envs)
    assert len(r['rows'])==6 and not r['full_protocol']
    if fault=='step_raise':assert r['counts']['public_step']['attempted']>r['counts']['public_step']['completed']
    if fault=='render_raise':assert 'primary render failure' in r['first_error']['message']


def test_reference_read_failure_still_saves_new_image_and_receipt(monkeypatch):
    data,images=reference_fixture()
    monkeypatch.setattr(m,'verify',lambda p:deepcopy(data));monkeypatch.setattr(m.moving,'validate_env',lambda _:None)
    monkeypatch.setattr(m.moving.source,'sample_state',lambda *a:REQUEST.copy())
    def fail(r):raise OSError('old PNG disappeared')
    monkeypatch.setattr(m,'old_image',fail)
    s=RAMStore();r=m.execute({},s,Env,authorization=m.AUTH)
    e=next(e for e in s.events if e['kind']=='fresh_overlay_frame')
    assert e['receipt']['png'] in s.files and e['receipt']['reference_error']['type']=='OSError'
    assert r['status']=='engineering_failed' and 'old PNG disappeared' in r['first_error']['message']


def test_journal_fsync_failure_is_not_silent_success(monkeypatch):
    r,s,envs,_=run(monkeypatch,store=RAMStore(fault='fresh_overlay_frame'))
    assert r['status']=='engineering_failed' and 'fsync' in r['first_error']['message']
    assert r['blind_exports']==[] and any(n.endswith('.png') for n in s.files)


def test_no_old_execute_episode_or_plan_functions_called(monkeypatch):
    monkeypatch.setattr(m.moving,'episode',lambda *a:pytest.fail('old episode called'))
    monkeypatch.setattr(m.moving,'execute',lambda *a,**kw:pytest.fail('old source execute called'))
    monkeypatch.setattr(m.previous,'execute',lambda *a,**kw:pytest.fail('old physical execute called'))
    monkeypatch.setattr(m.previous,'branch',lambda *a:pytest.fail('old plan branch called'))
    r,_,_,_=run(monkeypatch);assert r['full_protocol']


def test_no_authority_no_source_read_or_env(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda p:pytest.fail('unauthorized read'))
    with pytest.raises(ValueError):m.execute({},RAMStore(),lambda:pytest.fail('factory'),authorization='no')


def test_meter_failure_cap_attempt_return_and_no_redispatch():
    s=RAMStore();meter=m.Meter(s);meter.p=deepcopy(meter.p);meter.p['caps']['explicit_render']=1
    def fail():raise RuntimeError('dispatch failed')
    with pytest.raises(RuntimeError):meter.call('explicit_render',fail)
    assert meter.counts['explicit_render']==dict(attempted=1,completed=0)
    with pytest.raises(ValueError):meter.call('explicit_render',lambda:pytest.fail('cap exceeded'))


def test_build_verify_binds_whole_source_reference_index_and_three_files(monkeypatch):
    data,_=reference_fixture();data['bindings']={'/RAM/old.py':'a'*64}
    monkeypatch.setattr(m,'inputs',lambda:deepcopy(data));monkeypatch.setattr(m,'sha',lambda p:'a'*64)
    p=m.build();assert m.verify(p)['refs'][0]['root_sha256']==data['refs'][0]['root_sha256']
    assert len(p['bindings'])==4 and p['reference_index_sha256']==m.digest(data['references'])
    # Caller must not mutate PINNED via aliasing the new prepare.
    before=deepcopy(m.PINNED);p['old_inputs']['extra']='bad';assert m.PINNED==before
    with pytest.raises(ValueError):m.verify(p)
    p=m.build();p['contract']['caps']['public_step']=121
    with pytest.raises(ValueError):m.verify(p)


def test_final_report_failure_never_marks_complete(monkeypatch):
    class BadFinal(RAMStore):
        def put(self,n,v,final=False):
            if final:raise OSError('mock report fsync failure')
            return super().put(n,v,final)
    r,_,_,_=run(monkeypatch,store=BadFinal())
    assert r['status']=='engineering_failed' and not r['full_protocol'] and 'report fsync' in r['first_error']['message']
