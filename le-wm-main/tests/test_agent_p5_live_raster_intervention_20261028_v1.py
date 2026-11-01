"""All output in RAM; no native solver, environment, model or scientific result."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import random
from types import SimpleNamespace as NS

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p5_live_raster_intervention_20261028_v1 as m


@pytest.fixture
def ram(monkeypatch):
    from PIL import Image
    from experiments.spectral_reachability import p5_jepa_factual_pilot_20261027_v1 as raw
    memory={};faults={};original_colors={}
    class Body:
        def __init__(self): self.position=np.zeros(2,np.float64)
    class Space:
        def __init__(self): self.bodies=[];self.shapes=[]
        def step(self,dt): assert dt==.01  # fixture counter only, no solver
        def debug_draw(self,x): return None
    class Env:
        def __init__(self,role):self.role=role;self.seed(None)
        def seed(self,seed=None):self.seedval=seed
        def _setup(self):
            self.space=Space();self.agent=Body();self.block=Body();self.static=Body()
            self.space.bodies=[self.agent,self.block,self.static];self.space.shapes=[object()]
            self.goal_color=list(m.h.RGBA['visible']);original_colors[self.role]=self.goal_color;self.n=0
        def _set_state(self):self.space.step(.01)
        def _get_obs(self):return np.asarray([*self.agent.position,0.,0.,0.,0.,0.],np.float64)
        def _get_goal_pose_body(self,pose):return Body()
        def _render_frame(self,mode):
            self._get_goal_pose_body(None);self.space.debug_draw(None)
            if faults.get('raise_render') and self.n==15:raise RuntimeError('fixture no returned frame')
            if faults.get('color_raise') and self.goal_color[:3]==[255,255,255]:raise RuntimeError('fixture color has no returned frame')
            a=np.zeros((224,224,3),np.uint8);a[0,0]=self.goal_color[:3]
            if self.n>=11 and not faults.get('cache_missing'):a[120:123,88]=[255,0,0]
            if faults.get('mutate') and self.n==15 and self.goal_color[:3]==[255,255,255]:self.agent.position[0]+=1
            if faults.get('rng') and self.n==15 and self.goal_color[:3]==[255,255,255]:random.random()
            return a
        def reset(self):
            self._setup();self._set_state();state=self._get_obs()
            return dict(visual=self._render_frame('rgb_array'),proprio=state[[0,1,5,6]]),state
        def step(self,u):
            for i in range(10):self.space.step(.01)
            self.n+=1;self.agent.position+=u;self._get_goal_pose_body(None);state=self._get_obs()
            done=bool(faults.get('early') and self.n==12)
            return dict(visual=self._render_frame('rgb_array'),proprio=state[[0,1,5,6]]),0.,done,dict(state=state)
        def close(self):return None
    def public(env):return dict(role=env.role,frame=env.n,rng={'seed':env.seedval},
        bodies=dict(agent=dict(position=env.agent.position.tolist())),space=dict(current_time_step=.01),
        env=dict(goal_pose=[256.,256.,.7],coverage_arr=[float(env.n)]))
    base=NS(public_state=public,numeric_json=raw.numeric_json)
    p=dict(schema=m.NAME+'.prepare',protocol=m.protocol(),plans=[],frames={},targets=[],
        v3_prepare='/source/v3',v3_sha256=None,goal_protocol=dict(constructor=dict(role='goal'),seed=99),
        old_failure=dict(path='/source/failed',sha256=m.FAILED_SHA))
    for role in m.STATES:
        plan=dict(arm=role,constructor=dict(role=role),seed=99,prefix_raw_u=[[0.,0.]]*10,
            reference_suffix_raw_u=[[.1,0.]]*5+[[0.,.1]]*5)
        if role!='goal':p['plans'].append(plan)
        env=Env(role);env.seed(99);obs,state=env.reset();records=[];p['frames'][role]=records
        def source(n,obs,state,transition):
            vals={'rgb.npy':m.h.npy(obs['visual']),'state.npy':m.h.npy(state),'proprio.npy':m.h.npy(obs['proprio']),
                'transition.json':m.canonical(raw.numeric_json(transition))+b'\n','full_public_state.json':m.canonical(public(env))+b'\n'}
            rec={}
            for name,b in vals.items():
                path='/source/'+role+f'/{n}/'+name;memory[path]=b
                rec[name]=dict(path=path,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
            records.append(rec)
        source(0,obs,state,dict(kind='independent_goal_scene_reset' if role=='goal' else 'reset'))
        if role!='goal':
            for n,u in enumerate(plan['prefix_raw_u']+plan['reference_suffix_raw_u'][:5],1):
                obs,reward,done,info=env.step(np.asarray(u,np.float64));source(n,obs,info['state'],dict(kind='step',reward=reward,done=done,info=info))
        p['targets'].append(dict(state_id=role,frame=0 if role=='goal' else 15,source_public=records[-1]['full_public_state.json'],source_rgb=records[-1]['rgb.npy']))
    memory['/source/v3']=m.canonical(dict(origins=dict(environment='e',pymunk='p',pygame='g')))
    class Buffer(io.BytesIO):
        def close(self):return None
    class RAMStore(m.Store):
        def __init__(self,out):
            self.out=Path(out);self.started=m.time.monotonic();self.used=0;self.files={};self.previous=None;self.sequence=0;self.journal=Buffer()
        def save(self,name,b,final=False):
            path=str(self.out/name);assert path not in memory
            memory[path]=b;self.used+=len(b)
            rec=dict(path=name,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b));self.files[name]=rec;return rec
        def event(self,event,final=False):
            row=dict(sequence=self.sequence,previous_sha256=self.previous,event=event);row['entry_sha256']=m.digest(row)
            b=m.canonical(row)+b'\n';self.journal.write(b);memory[str(self.out/'journal.jsonl')]=self.journal.getvalue()
            self.used+=len(b);self.sequence+=1;self.previous=row['entry_sha256']
    oldread,oldstat,oldfile,oldglob=Path.read_bytes,Path.stat,Path.is_file,Path.rglob
    oldsha,oldload,oldimage,oldreader=m.sha,np.load,Image.open,m.read
    monkeypatch.setattr(Path,'read_bytes',lambda p:memory[str(p)] if str(p) in memory else oldread(p))
    monkeypatch.setattr(Path,'stat',lambda p,*a,**k:NS(st_size=len(memory[str(p)]),st_mode=0o100644) if str(p) in memory else oldstat(p,*a,**k))
    monkeypatch.setattr(Path,'is_file',lambda p:str(p) in memory or oldfile(p))
    monkeypatch.setattr(Path,'rglob',lambda p,pattern:(Path(x) for x in list(memory) if x.startswith(str(p)+'/')) if str(p)=='/ram-raster' else oldglob(p,pattern))
    monkeypatch.setattr(m,'sha',lambda p:hashlib.sha256(memory[str(p)]).hexdigest() if str(p) in memory else oldsha(p))
    def reader(path,expected=None):
        if str(path) not in memory:return oldreader(path,expected)
        if expected is not None:assert m.sha(path)==expected
        return json.loads(memory[str(path)])
    monkeypatch.setattr(m,'read',reader)
    monkeypatch.setattr(np,'load',lambda p,*a,**k:oldload(io.BytesIO(memory[str(p)]) if str(p) in memory else p,*a,**k))
    monkeypatch.setattr(Image,'open',lambda p,*a,**k:oldimage(io.BytesIO(memory[str(p)]) if str(p) in memory else p,*a,**k))
    monkeypatch.setattr(m,'Store',RAMStore);monkeypatch.setattr(m,'verify',lambda *a:p)
    # Pure saved origin validator replaced in fixture; no runtime dependency load.
    monkeypatch.setattr(m.h,'load_bound',lambda *a:NS(check_origins=lambda *a:None))
    loader=lambda:(Env,NS(Space=Space,Body=Body),NS(Color=lambda name:list(m.h.RGBA['visible' if name=='LightGreen' else 'white'])),base,dict(fixture=True))
    return p,memory,loader,faults,original_colors


def execute(ram):
    p,memory,loader,_,_=ram
    r=m.execute(p,'fixture-prepare','/ram-raster',loader)
    return r


def consume():return m.consume('/unused','fixture-prepare','/ram-raster/report.json',m.sha('/ram-raster/report.json'))


def test_complete_fixture_execute_consume(ram):
    r=execute(ram)
    assert r['errors']==[] and r['meter']['entries']==m.EXPECTED
    receipt=consume()
    assert receipt['status']=='strict_live_raster_complete'
    assert len(receipt['encoder_inputs'])==10 and len(receipt['source_comparisons'])==65
    assert receipt['old_renderer_release'] is False and receipt['historical_full_public_equivalence'] is None
    assert [x['image_key'] for x in receipt['encoder_inputs']]==[s+'/'+c for s in m.STATES for c in m.COLORS]
    for x in receipt['encoder_inputs']:
        after=m.read('/ram-raster/'+x['after']['path'])
        assert x['fresh_snapshot_digest']==m.digest(after)
        assert x['fresh_public_state_digest']==m.digest(after['public_state'])
    assert len([k for k in ram[1] if k.startswith('/ram-raster/') and k.endswith('.png')])==85


@pytest.mark.parametrize('fault',['mutate','rng','raise_render','color_raise','cache_missing','early'])
def test_failures_preserve_fixed_rows_slots(ram,fault):
    original_rng=random.getstate();ram[3][fault]=True
    try:r=execute(ram)
    finally:random.setstate(original_rng)
    assert r['errors'] and r['status']=='partial_engineering_failure'
    assert len(r['rows'])==5 and sum(len(x['cells']) for x in r['rows'])==10
    assert sum(len(x['shots']) for x in r['rows'])==25
    assert not r['old_renderer_release']
    with pytest.raises(ValueError):consume()


def test_suffix_full_public_diff_allowed_but_not_repaired(ram):
    p,memory,*_=ram
    rec=p['frames']['xy'][13]['full_public_state.json'];v=m.read(rec['path'])
    v['env']['coverage_arr'][0]=float(np.nextafter(13.,float('inf')))
    b=m.canonical(v)+b'\n';memory[rec['path']]=b;rec['sha256']=hashlib.sha256(b).hexdigest();rec['bytes']=len(b)
    r=execute(ram);assert not r['errors']
    exact=r['rows'][0]['records'][13]['exact'];assert not exact['full_public_state.json']
    receipt=consume();comparison=receipt['source_comparisons'][13]
    diff=m.read('/ram-raster/'+comparison['differences']['path'])
    assert diff['full_public_state.json'][0]['pointer']=='/env/coverage_arr/0'
    assert receipt['historical_full_public_equivalence'] is None


def test_prefix_full_public_mismatch_stops_before_suffix(ram):
    p,memory,*_=ram;rec=p['frames']['yx'][9]['full_public_state.json']
    memory[rec['path']]=m.canonical(dict(wrong=True))+b'\n'
    r=execute(ram)
    assert r['errors'] and not r['all_states_attained']
    assert '/ram-raster/yx/frame_09/full_public_state.json' in memory
    assert '/ram-raster/xy/command_11.json' not in memory


@pytest.mark.parametrize('mutation',['missing_row','missing_shot','baseline','png','diff','restoration','color','budget','raw'])
def test_saved_mutations_failclosed(ram,mutation):
    r=execute(ram);assert not r['errors'];memory=ram[1]
    if mutation=='missing_row':r['rows'].pop()
    elif mutation=='missing_shot':r['rows'][0]['shots'].pop()
    elif mutation=='baseline':r['rows'][0]['baseline']=r['rows'][0]['records'][0]
    elif mutation=='color':r['rows'][0]['cells'][0]['requested_rgba']=[0,0,0,0]
    elif mutation=='budget':r['meter']['entries']['space_step']=0
    else:
        path={'png':'xy/white.png','diff':'xy/frame_13/source_differences.json','restoration':'xy/white_restoration.json','raw':'xy/command_11.json'}[mutation]
        memory['/ram-raster/'+path]+=b' '
    memory['/ram-raster/report.json']=m.canonical(r)+b'\n'
    with pytest.raises((ValueError,KeyError)):consume()


def test_render_invariance_failure_frame_is_durable(ram):
    ram[3]['mutate']=True;r=execute(ram)
    assert r['errors']
    memory=ram[1]
    assert all('/ram-raster/xy/'+x in memory for x in ('white.npy','white.png','white_before.json','white_after.json','white_restoration.json'))
    assert m.read('/ram-raster/xy/white_restoration.json')['same_color_object']


def test_internal_color_exception_does_not_reuse_old_observation(ram):
    ram[3]['color_raise']=True;r=execute(ram);memory=ram[1]
    assert r['errors'] and '/ram-raster/xy/white_restoration.json' in memory
    assert '/ram-raster/xy/white.npy' not in memory and '/ram-raster/xy/white_after.json' not in memory
    receipt=m.read('/ram-raster/xy/white_restoration.json')
    assert receipt['same_color_object'] and receipt['snapshot']['goal_color']['rgba']==m.h.RGBA['visible']


def test_load_failure_has_no_fake_observations(ram):
    def broken():raise ImportError('fixture unavailable')
    r=m.execute(ram[0],'fixture-prepare','/ram-raster',broken)
    assert r['errors'] and r['meter'] is None
    assert all(not row['records'] for row in r['rows'])


def test_budget_original_helper_unmodified():
    assert m.EXPECTED['space_step']==605 and m.h.EXPECTED['space_step']==805
    assert m.EXPECTED['native_render']==65+20==85
    assert m.EXPECTED['native_goal_body']==5+60*2+20
    assert m.protocol()['images']['intervention']==10


def test_no_authorization_no_runtime(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('no read before auth'))
    with pytest.raises(ValueError,match='authorization'):m.run(None,None,None,False)


def test_diff_exact_signed_zero_missing_and_oneulp():
    assert m.differences(-0.,0.)
    assert m.differences(1.,float(np.nextafter(1.,2.)))
    d=m.differences({'x':0},{'y':0})
    assert len(d)==2 and d[0]['actual_present'] is False
    assert m.differences({'a':[1,2]},{'a':[1,3]})[0]['pointer']=='/a/1'


def test_snapshot_separates_color_and_public_state(ram):
    _,_,loader,_,_=ram;cls,_,_,base,_=loader();env=cls('xy');env.reset()
    before=m.snapshot(env,base);env.goal_color=[255,255,255,255];after=m.snapshot(env,base)
    assert before['public_state']==after['public_state']
    assert not m.equal(before,after)
