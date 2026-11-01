"""RAM-only control-flow fixtures. No physics/native runtime imports or model calls."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p5_replay_target_renderer_20261028_v1 as m


class RAMStore:
    def __init__(self):
        self.data = {}; self.events = []

    def save(self, path, b, final=False):
        assert path not in self.data
        self.data[path] = b
        return dict(path=path, bytes=len(b), sha256=hashlib.sha256(b).hexdigest())

    def json(self, path, obj, final=False):
        return self.save(path, m.canonical(obj)+b'\n', final)

    def image(self, path, a):
        return dict(npy=self.save(path+'.npy', m.npy(a)),
                    png=self.save(path+'.png', a.tobytes()),
                    pixel_sha256=hashlib.sha256(a.tobytes()).hexdigest())

    def event(self, event, final=False):
        self.events.append(deepcopy(event))


class FakeEnv:
    def __init__(self, cache=True):
        self.goal_color = [144,238,144,255]
        self.agent = object(); self.block = object()
        self.space = NS(bodies=[self.agent,self.block], shapes=[object()])
        self.cache = cache; self.value = 1; self.fail = False; self.mutate = False

    def _render_frame(self, mode):
        assert mode == 'rgb_array'
        if self.fail:
            raise RuntimeError('render failure has no returned frame')
        a = np.zeros((224,224,3), np.uint8)
        # Three contact dots exist only with live collision cache, not public snapshot.
        if self.cache:
            a[120:123,88] = [255,0,0]
        a[0,0] = self.goal_color[:3]
        if self.mutate:
            self.value += 1
        return a


def public(env):
    return dict(value=env.value, rng={'fixed':7}, env={'goal_pose':[256,256,np.pi/4]},
                space={'current_time_step':.01})


@pytest.fixture
def rendering():
    store = RAMStore(); env = FakeEnv()
    meter = NS(errors=[], context=None)
    return env, store, meter, NS(Color=lambda name: list(m.RGBA['visible' if name == 'LightGreen' else 'white'])), NS(public_state=public)


def plans():
    return [dict(arm=k,prefix_raw_u=[[0.,0.]]*10,reference_suffix_raw_u=[[.1,0.]]*10) for k in m.STATES[:4]]


def states():
    return [dict(state_id=k) for k in m.STATES]


def test_exact_fixed_slots():
    m.validate_slots(states(), plans())
    assert m.protocol()['denominator'] == 10
    assert m.protocol()['baseline_denominator'] == 5
    assert m.EXPECTED['space_step'] == 5+80*10 == 805
    assert m.EXPECTED['native_render'] == 85+10
    assert m.EXPECTED['native_goal_body'] == 4*(40+1)+1+1+10
    assert m.EXPECTED['body_construct'] == 5*3+m.EXPECTED['native_goal_body']


@pytest.mark.parametrize('which',range(5))
def test_missing_state_rejected(which):
    s = states(); s.pop(which)
    with pytest.raises(ValueError,match='five states'):
        m.validate_slots(s,plans())


@pytest.mark.parametrize('change',('missing','duplicate','reordered','short','nonfinite'))
def test_bad_plan_denominator_or_history(change):
    p = plans()
    if change == 'missing': p.pop()
    if change == 'duplicate': p[1] = p[0]
    if change == 'reordered': p.reverse()
    if change == 'short': p[0]['prefix_raw_u'] = [[0.,0.]]*9
    if change == 'nonfinite': p[0]['prefix_raw_u'] = [[float('nan'),0.]]*10
    with pytest.raises(ValueError): m.validate_slots(states(),p)


def test_contact_cache_seam_is_not_in_public_state(rendering):
    live,store,meter,pygame,base = rendering
    hydrated = FakeEnv(cache=False)
    assert public(live) == public(hydrated)
    a = live._render_frame('rgb_array'); b = hydrated._render_frame('rgb_array')
    ys,xs = np.where(np.any(a != b,axis=2))
    assert len(ys) == 3 and [xs.min(),ys.min(),xs.max(),ys.max()] == [88,120,88,122]
    cell = dict(color='visible',status='not_run')
    m.render_color(live,'visible',pygame,base,store,meter,'yx',cell)
    actual = np.load(io.BytesIO(store.data['yx/visible.npy']),allow_pickle=False)
    assert np.array_equal(actual,a) and not np.array_equal(actual,b)


@pytest.mark.parametrize('label',m.COLORS)
def test_native_color_restores_same_object_and_state(rendering,label):
    env,store,meter,pygame,base = rendering
    old = env.goal_color; before = public(env); cell = dict(color=label,status='not_run')
    m.render_color(env,label,pygame,base,store,meter,'xy',cell)
    assert env.goal_color is old and public(env) == before
    restored = json.loads(store.data['xy/'+label+'_restored.json'])
    assert restored['same_color_object'] and restored['topology_unchanged']
    assert restored['public_state']['space']['current_time_step'] == .01
    assert cell['status'] == 'rendered'
    assert [e['phase'] for e in store.events] == ['intent','returned']


def test_failing_invariance_frame_is_durable_before_stop(rendering):
    env,store,meter,pygame,base = rendering; env.mutate = True
    old = env.goal_color; cell = dict(color='white',status='not_run')
    with pytest.raises(ValueError,match='state/RNG/topology'):
        m.render_color(env,'white',pygame,base,store,meter,'yx',cell)
    assert 'yx/white.npy' in store.data and 'yx/white_after.json' in store.data
    assert 'yx/white_restored.json' in store.data and env.goal_color is old
    assert cell['status'] != 'rendered'
    assert json.loads(store.data['yx/white_before.json']) != json.loads(store.data['yx/white_after.json'])


def test_render_exception_does_not_claim_old_frame(rendering):
    env,store,meter,pygame,base = rendering; env.fail = True
    old = env.goal_color; cell = dict(color='white',status='not_run')
    with pytest.raises(RuntimeError):
        m.render_color(env,'white',pygame,base,store,meter,'yx',cell)
    assert 'image' not in cell and 'yx/white.npy' not in store.data
    assert env.goal_color is old and 'yx/white_restored.json' in store.data
    assert [e['phase'] for e in store.events] == ['intent','failed']


def test_byte_gate_no_pixel_tolerance(monkeypatch):
    files = {'/a':b'RGB-contact-red','/b':b'RGB-contact-red','/wrong':b'RGB-contact-Red'}
    monkeypatch.setattr(Path,'read_bytes',lambda p: files[str(p)])
    monkeypatch.setattr(m,'sha',lambda p: hashlib.sha256(files[str(p)]).hexdigest())
    expected = dict(path='/a',sha256=m.sha('/a'))
    m.byte_gate('/b',expected)
    with pytest.raises(ValueError,match='bytes'): m.byte_gate('/wrong',expected)


def test_wrong_baseline_sha_rejected_even_identical_pixels(monkeypatch):
    monkeypatch.setattr(Path,'read_bytes',lambda p:b'same')
    monkeypatch.setattr(m,'sha',lambda p:'actual')
    with pytest.raises(ValueError): m.byte_gate('/b',dict(path='/a',sha256='wrong'))


def test_profile_counts_actual_entries_returns_and_dt():
    def f(self): return None
    def step(self,dt): return None
    # Distinct code objects are necessary in real runtime; fixture only tests the
    # specifically observed step event, not a claim about native execution.
    cls = NS(**{k:f for k in ('__init__','seed','reset','step','_get_obs','_render_frame','_setup','_set_state','_get_goal_pose_body','close')})
    py = NS(Space=NS(__init__=f,step=step,debug_draw=f),Body=NS(__init__=f))
    s = RAMStore(); meter = m.Meter(cls,py,s)
    frame = NS(f_code=step.__code__,f_locals={'dt':.01})
    meter.context = dict(state_id='xy',operation='step',frame=1)
    meter.profile(frame,'call',None); meter.profile(frame,'return',None)
    assert meter.receipt()['entries']['space_step'] == 1
    assert meter.receipt()['return_events']['space_step'] == 1
    assert s.events[0]['dt'] == .01 and s.events[0]['context']['frame'] == 1


def test_profiler_io_failure_records_error_without_throwing():
    def f(self): return None
    cls = NS(**{k:f for k in ('__init__','seed','reset','step','_get_obs','_render_frame','_setup','_set_state','_get_goal_pose_body','close')})
    py = NS(Space=NS(__init__=f,step=f,debug_draw=f),Body=NS(__init__=f))
    class FailedStore:
        def event(self,*a,**k): raise OSError('disk failure')
    meter = m.Meter(cls,py,FailedStore())
    meter.profile(NS(f_code=f.__code__,f_locals={}),'call',None)
    assert meter.errors == ['OSError: disk failure']


def test_operation_refuses_observer_error_before_new_dispatch():
    called = []; s = RAMStore(); meter = NS(errors=['broken'],context=None)
    with pytest.raises(ValueError,match='observer'):
        m.operation(s,meter,'xy','step',{},lambda:called.append(True))
    assert not called and not s.events


def test_cleanup_does_not_skip_due_to_prior_observer_error():
    s = RAMStore(); meter = NS(errors=['broken'],context=None)
    assert m.operation(s,meter,'xy','close',{},lambda:42,cleanup=True) == 42
    assert [e['phase'] for e in s.events] == ['intent','returned']


def test_no_physics_authorization_no_verify_or_loader(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('must not read/load before authorization'))
    with pytest.raises(ValueError,match='MAIN explicit'):
        m.run('/unused','unused','/unused',False)


def test_npy_preserves_float64_signed_zero():
    a = np.array([[.1,0.],[-.1,-0.]],np.float64)
    b = np.load(io.BytesIO(m.npy(a)),allow_pickle=False)
    assert a.dtype == b.dtype and a.tobytes() == b.tobytes()


def test_prepare_bound_to_actual_v3_not_failed_static_protocol():
    assert m.V3_PREP_SHA == 'fdf76d7f0d404065bdee5abd404d4f39e0f68877263cc4862ce2dc84fe20109a'
    assert m.protocol()['cache'].startswith('original live')
    assert m.protocol()['physics_steps'] == 805
    assert not m.protocol()['scientific_admission']


def test_journal_hash_includes_order_and_payload():
    a = dict(sequence=0,previous_sha256=None,event={'phase':'intent','operation':'step'})
    b = deepcopy(a); b['event']['operation'] = 'reset'
    assert m.digest(a) != m.digest(b)


def test_duplicate_json_keys_refused(monkeypatch):
    monkeypatch.setattr(m,'sha',lambda p:'hash')
    monkeypatch.setattr(Path,'stat',lambda p:NS(st_size=20))
    monkeypatch.setattr(Path,'read_bytes',lambda p:b'{"x":1,"x":2}')
    with pytest.raises(ValueError,match='duplicate'):
        m.read('/ram')


@pytest.fixture
def full_ram(monkeypatch):
    """Exercise execute->consume against RAM files and nonphysical Python fakes.

    Every method below is a structural fixture, NOT native physics or evidence.
    No simulator import, Space solver, model, disk result or future SHA is used.
    """
    from PIL import Image
    from experiments.spectral_reachability import p5_jepa_factual_pilot_20261027_v1 as raw
    memory = {}
    class Body:
        def __init__(self): self.position = np.zeros(2,np.float64)
    class Space:
        def __init__(self): self.bodies = []; self.shapes = []
        def step(self,dt): assert dt == .01  # no solver
        def debug_draw(self,options): return None
    class Env:
        def __init__(self,role): self.role = role; self.seed(None)
        def seed(self,seed=None): self.seedval = seed
        def _setup(self):
            self.space = Space(); self.agent = Body(); self.block = Body(); self.static = Body()
            self.space.bodies = [self.agent,self.block,self.static]; self.space.shapes = [object()]
            self.goal_color = list(m.RGBA['visible']); self.n = 0
        def _set_state(self): self.space.step(.01)
        def _get_obs(self):
            return np.asarray([*self.agent.position,0.,0.,0.,0.,0.],np.float32)
        def _get_goal_pose_body(self,pose): return Body()
        def _render_frame(self,mode):
            self._get_goal_pose_body(None); self.space.debug_draw(None)
            a = np.zeros((224,224,3),np.uint8); a[0,0] = self.goal_color[:3]
            if self.n >= 11: a[120:123,88] = [255,0,0]
            return a
        def reset(self):
            self._setup(); self._set_state(); state = self._get_obs()
            return dict(visual=self._render_frame('rgb_array'),proprio=state[[0,1,5,6]]),state
        def step(self,u):
            for _ in range(10): self.space.step(.01)
            self.n += 1; self.agent.position += u
            self._get_goal_pose_body(None); state = self._get_obs()
            return dict(visual=self._render_frame('rgb_array'),proprio=state[[0,1,5,6]]),0.,False,dict(state=state)
        def close(self): return None
    def pub(env):
        return dict(role=env.role,frame=env.n,rng={'seed':env.seedval},
                    bodies={'agent':{'position':env.agent.position.tolist()}},space={'current_time_step':.01})
    base = NS(public_state=pub,rng_state=lambda e:dict(seed=e.seedval),numeric_json=raw.numeric_json)
    def geometry(env): env._get_goal_pose_body(None); return dict(fixture_only=True)
    p = dict(protocol=m.protocol(),plans=[],frames={},states=[],goal_protocol={'constructor':{'role':'goal'},'seed':99},v3_prepare='/source/v3',v3_sha256=None)
    for role in m.STATES:
        plan = dict(arm=role,constructor={'role':role},seed=99,prefix_raw_u=[[0.,0.]]*10,reference_suffix_raw_u=[[.1,0.]]*10)
        if role != 'goal': p['plans'].append(plan)
        env = Env(role=role); env.seed(99); obs,state = env.reset()
        records = []; p['frames'][role] = records
        def source_frame(n,obs,state,transition):
            values = {'rgb.npy':m.npy(obs['visual']),'state.npy':m.npy(state),'proprio.npy':m.npy(obs['proprio']),
                      'transition.json':m.canonical(raw.numeric_json(transition))+b'\n','full_public_state.json':m.canonical(pub(env))+b'\n'}
            record = {}
            for name,b in values.items():
                path = '/source/'+role+f'/{n}/'+name; memory[path] = b
                record[name] = dict(path=path,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
            records.append(record)
        source_frame(0,obs,state,dict(kind='independent_goal_scene_reset' if role == 'goal' else 'reset'))
        if role != 'goal':
            for n,u in enumerate(plan['prefix_raw_u']+plan['reference_suffix_raw_u'],1):
                obs,reward,done,info = env.step(np.asarray(u,np.float64))
                source_frame(n,obs,info['state'],dict(kind='step',reward=reward,done=done,info=info))
        p['states'].append(dict(state_id=role,source_state=pub(env),rgb=records[-1]['rgb.npy']))
    memory['/source/geometry'] = m.canonical({'fixture_only':True})+b'\n'
    p['goal_geometry'] = dict(path='/source/geometry',sha256=hashlib.sha256(memory['/source/geometry']).hexdigest())
    memory['/source/v3'] = m.canonical({'origins':dict(environment='e',pymunk='p',pygame='g')})
    class Buffer(io.BytesIO):
        def close(self): return None
    class RAMOutput(m.Store):
        def __init__(self,out):
            self.out = Path(out); self.started = m.time.monotonic(); self.used = 0
            self.previous = None; self.sequence = 0; self.files = {}; self.journal = Buffer()
        def save(self,name,b,final=False):
            key = str(self.out/name); assert key not in memory
            memory[key] = b; self.used += len(b)
            rec = dict(path=name,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)); self.files[name] = rec
            return rec
        def event(self,event,final=False):
            row = dict(sequence=self.sequence,previous_sha256=self.previous,event=event)
            row['entry_sha256'] = m.digest(row); b = m.canonical(row)+b'\n'
            self.journal.write(b); memory[str(self.out/'journal.jsonl')] = self.journal.getvalue()
            self.used += len(b); self.sequence += 1; self.previous = row['entry_sha256']
    old_read,old_stat,old_isfile,old_rglob = Path.read_bytes,Path.stat,Path.is_file,Path.rglob
    old_sha,old_load,old_open = m.sha,np.load,Image.open
    monkeypatch.setattr(Path,'read_bytes',lambda x:memory[str(x)] if str(x) in memory else old_read(x))
    monkeypatch.setattr(Path,'stat',lambda x,*a,**k:NS(st_size=len(memory[str(x)]),st_mode=0o100644) if str(x) in memory else old_stat(x,*a,**k))
    monkeypatch.setattr(Path,'is_file',lambda x:str(x) in memory or old_isfile(x))
    monkeypatch.setattr(Path,'rglob',lambda x,pattern:(Path(k) for k in list(memory) if k.startswith(str(x)+'/')) if str(x)=='/ram-output' else old_rglob(x,pattern))
    monkeypatch.setattr(m,'sha',lambda x:hashlib.sha256(memory[str(x)]).hexdigest() if str(x) in memory else old_sha(x))
    monkeypatch.setattr(np,'load',lambda x,*a,**k:old_load(io.BytesIO(memory[str(x)]) if str(x) in memory else x,*a,**k))
    monkeypatch.setattr(Image,'open',lambda x,*a,**k:old_open(io.BytesIO(memory[str(x)]) if str(x) in memory else x,*a,**k))
    monkeypatch.setattr(m,'Store',RAMOutput)
    monkeypatch.setattr(m,'verify',lambda *a:p)
    monkeypatch.setattr(m,'load_bound',lambda name,expected:NS(geometry_witness=geometry) if name==m.SCENE else NS(check_origins=lambda *a:None))
    loader = lambda:(Env,NS(Space=Space,Body=Body),NS(Color=lambda name:list(m.RGBA['visible' if name=='LightGreen' else 'white'])),base,{'fixture':True})
    return p,memory,loader


def test_complete_ram_execute_and_strict_consume(full_ram):
    p,memory,loader = full_ram
    r = m.execute(p,'fixture-prepare-sha','/ram-output',loader)
    assert r['errors'] == []
    assert r['meter']['entries'] == m.EXPECTED
    assert len(r['rows']) == 5 and all(len(row['shots'])==3 for row in r['rows'])
    assert len([k for k in memory if k.startswith('/ram-output/') and k.endswith('.png')]) == 95
    result = m.consume('/unused','fixture-prepare-sha','/ram-output/report.json',m.sha('/ram-output/report.json'))
    assert result['status'] == 'strict_saved_complete' and not result['scientific_admission']


@pytest.mark.parametrize('mutation',('baseline','missing_slot','png'))
def test_full_saved_mutations_fail_closed(full_ram,mutation):
    p,memory,loader = full_ram
    r = m.execute(p,'fixture-prepare-sha','/ram-output',loader)
    assert r['errors'] == []
    if mutation == 'missing_slot':
        r['rows'].pop()
    elif mutation == 'baseline':
        r['rows'][0]['baseline'] = r['rows'][0]['records'][0]
    else:
        memory['/ram-output/xy/white.png'] += b'trailing-tamper'
    memory['/ram-output/report.json'] = m.canonical(r)+b'\n'
    with pytest.raises(ValueError):
        m.consume('/unused','fixture-prepare-sha','/ram-output/report.json',m.sha('/ram-output/report.json'))


def test_replay_bad_baseline_preserves_failure_and_denominators(full_ram):
    p,memory,loader = full_ram
    p['frames']['yx'][20]['rgb.npy']['sha256'] = '0'*64
    r = m.execute(p,'fixture-prepare-sha','/ram-output',loader)
    assert r['status'] == 'partial_engineering_failure' and r['errors']
    assert len(r['rows']) == 5 and sum(len(x['cells']) for x in r['rows']) == 10
    assert all(c['status']=='not_run' for row in r['rows'] for c in row['cells'])
    assert not r['all_baselines_passed'] and r['meter']['entries']['native_render'] == 64
    assert '/ram-output/yx/frame_20/rgb.npy' in memory
    assert '/ram-output/yx/baseline_before.json' in memory and '/ram-output/yx/baseline_after.json' in memory
    assert r['meter']['entries']['close'] == 4
    assert '/ram-output/report.json' in memory
    with pytest.raises(ValueError,match='complete engineering'):
        m.consume('/unused','fixture-prepare-sha','/ram-output/report.json',m.sha('/ram-output/report.json'))


def test_runtime_load_failure_persists_full_planned_slots(full_ram):
    p,memory,_ = full_ram
    def broken(): raise ImportError('fixture runtime unavailable')
    r = m.execute(p,'fixture-prepare-sha','/ram-output',broken)
    assert r['status'] == 'partial_engineering_failure' and r['meter'] is None
    assert len(r['rows']) == 5 and sum(len(x['shots']) for x in r['rows']) == 15
    assert '/ram-output/report.json' in memory and all(not row['records'] for row in r['rows'])
