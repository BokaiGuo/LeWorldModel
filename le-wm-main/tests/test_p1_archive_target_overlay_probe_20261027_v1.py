"""RAM fake dynamics/render arrays only; no simulator, H5 data, CUDA or tmpdir."""
from copy import deepcopy
from io import BytesIO
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib
import numpy as np
from PIL import Image
import pytest
from experiments.spectral_reachability import p1_archive_target_overlay_probe_20261027_v1 as m

spec=importlib.util.spec_from_file_location('overlay_native_ram_fixture',Path(__file__).with_name('test_p1_archive_native_alignment_probe_20261027_v1.py'))
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


def png_bytes(a):
    s=BytesIO();Image.fromarray(a).save(s,format='PNG');return s.getvalue()


class MemoryStore(h.MemoryStore):
    def frame(self,name,value):
        assert name not in self.files
        self.files[name]=png_bytes(value);return hashlib.sha256(self.files[name]).hexdigest()


class Env(h.FakeEnv):
    def __init__(self,*,mutate=None,early=False,close_error=False):
        super().__init__(close_error=close_error);self.render_mode='rgb_array';self.window_size=512
        self.goal_pose=np.array([256.,256.,0.]);self.variation_space=NS(value={'rendering':{'render_goal':1},
            'goal':{'position':np.array([256.,256.]),'angle':0.,'color':np.array([0,255,0],np.uint8)},
            'background':{'color':np.array([255]*3,np.uint8)}})
        self.mutate=mutate;self.early=early;self.on_seen=False;self.closed=False;self.render_flags=[]
    def render(self):
        a=super().render();self.render_flags.append(self.with_target)
        if self.with_target:
            a[20:40,20:40]=[0,255,0];self.on_seen=True
            if self.mutate=='state':self.state[0]+=1;self.agent.position=tuple(self.state[:2])
            elif self.mutate=='goalpose':self.goal_pose[0]+=1
            elif self.mutate=='color':self.variation_space.value['goal']['color'][0]=1
            elif self.mutate=='raise':raise RuntimeError('fake on render failure')
        elif self.on_seen and self.mutate=='replay':a[0,0]=[0,0,0]
        return a
    def eval_state(self,goal,state):
        # Native mock remains open unless explicit early termination requested.
        return bool(self.early and len(self.commands)>=1),float(np.linalg.norm(goal-state))
    def close(self):self.closed=True;super().close()


def reference_payload(ordinal):
    env=Env();_,info=env.reset(seed=0);goal=info['goal_state'].copy()
    initial=np.array([100.+(ordinal==6229),100.,150.,150.,0.,0.,0.],np.float32)
    obs,_=env.reset(seed=0,options=dict(state=initial,goal_state=goal))
    actions=np.tile(np.array([.01,.005],np.float32),(20,1));references=[];states=[];pixels=[]
    for step in range(21):
        if step:obs,*_=env.step(actions[step-1].copy())
        states.append(obs['state'].copy())
        if step in m.prior.CHECKPOINTS:
            a=env.render();pixels.append(a.copy());receipt=dict(step=step,png=f'old_e{ordinal}_{step}.png',
                png_sha256=hashlib.sha256(png_bytes(a)).hexdigest(),pixels=m.prior.array_id(a),
                state=obs['state'].tolist(),snapshot=m.base.snapshot(env))
            references.append(dict(receipt=receipt,pixels=a.copy()))
    return dict(actions=actions,initial_state=initial,comparison_states=np.stack(states),comparison_pixels=np.stack(pixels),references=references)


def execute(monkeypatch,env=None,change=None):
    payloads=[reference_payload(o) for o in m.prior.ORDINALS]
    if change:change(payloads)
    monkeypatch.setattr(m,'verify',lambda p:deepcopy(payloads))
    monkeypatch.setattr(m.prior,'validate_env',lambda b:None)
    env=Env() if env is None else env;store=MemoryStore()
    result=m.execute(dict(schema=m.NAME+'.prepare'),store,lambda:env,authorization=m.AUTH)
    return result,store,env,payloads


def test_full_two_lag0_replays_all_30png_exact_oldoff_and_counts(monkeypatch):
    r,s,e,p=execute(monkeypatch)
    assert r['status']=='overlay_probe_processed' and r['errors']==[] and r['fixed_denominator']==2 and r['expected_frames']==30
    assert r['counts']=={k:dict(attempted=n,completed=n) for k,n in m.contract()['caps'].items()}
    assert len(e.resets)==4 and len(e.commands)==40 and e.closed and e.with_target is False
    assert len([n for n in s.files if n.endswith('.png')])==30
    for row in r['rows']:
        assert row['status']=='overlay_full20_completed' and row['steps_completed']==20 and row['missing_checkpoints']==[]
        assert len(row['frames'])==15 and len(row['off_replay_checks'])==5
        for j,step in enumerate(m.prior.CHECKPOINTS):
            a,b,c=row['frames'][3*j:3*j+3]
            assert [(x['step'],x['overlay_index'],x['with_target']) for x in (a,b,c)]==[(step,0,False),(step,1,True),(step,2,False)]
            assert s.files[a['png']]==s.files[c['png']] and all(a['old_off_match'].values()) and all(c['old_off_match'].values())
            assert c['off_replay_match'] is True and b['old_off_match'] is None and b['off_replay_match'] is None
            assert a['pre']==a['post']==b['pre']==b['post']==c['pre']==c['post']
            assert row['off_replay_checks'][j]==dict(step=step,first_frame_index=3*j,last_frame_index=3*j+2,bit_exact=True)
    for i,c in enumerate(e.commands):assert m.base.exact(c,p[i//20]['actions'][i%20])
    # Hooks restore instance/class layout and per-reset space.step on all exits.
    assert all(k not in e.__dict__ for k in ('render','_set_state','_set_goal_state')) and 'step' not in e.__dict__
    assert len(m.raw(r))<512*1024 and sum(len(b) for b in s.files.values())+len(m.raw(s.events))<2*1024**2


@pytest.mark.parametrize('bad',['file','pixels','state','snapshot'])
def test_old_off_mismatch_frame_and_receipt_saved_before_reject(monkeypatch,bad):
    def change(ps):
        ref=ps[0]['references'][0]
        if bad=='file':ref['receipt']['png_sha256']='0'*64
        elif bad=='pixels':ref['pixels'][0,0]=0
        elif bad=='state':ref['receipt']['state'][0]+=1
        elif bad=='snapshot':ref['receipt']['snapshot']['bodies']['block']['float64_bits']='bad'
    r,s,e,_=execute(monkeypatch,change=change)
    assert r['status']=='engineering_failed' and r['first_error']['phase']=='execution'
    assert len(r['rows'][0]['frames'])==1 and r['rows'][1]['status']=='not_started'
    f=r['rows'][0]['frames'][0];assert f['png'] in s.files and not f['old_off_match'][bad]
    assert any(x['kind']=='fresh_overlay_frame' for x in s.events) and e.with_target is False


@pytest.mark.parametrize('bad',['state','goalpose','color','replay'])
def test_render_mutation_and_off_replay_retained_before_gate(monkeypatch,bad):
    r,s,e,_=execute(monkeypatch,Env(mutate=bad))
    assert r['status']=='engineering_failed' and r['rows'][1]['status']=='not_started'
    count=3 if bad=='replay' else 2
    assert len(r['rows'][0]['frames'])==count and len([n for n in s.files if n.endswith('.png')])==count
    assert e.with_target is False and e.closed
    if bad=='replay':assert r['rows'][0]['frames'][-1]['off_replay_match'] is False


def test_primary_render_error_survives_cleanup_and_attempt_count(monkeypatch):
    r,s,e,_=execute(monkeypatch,Env(mutate='raise',close_error=True))
    assert 'fake on render failure' in r['first_error']['message'] and any(x['phase']=='close' for x in r['errors'])
    assert r['counts']['explicit_render']==dict(attempted=2,completed=1)
    assert len(r['rows'][0]['frames'])==1 and e.with_target is False


def test_honor_early_native_stop_no_padding_all_two_rows_preserved(monkeypatch):
    r,s,e,_=execute(monkeypatch,Env(early=True))
    assert r['status']=='overlay_probe_partial_native_stop' and r['errors']==[]
    assert [x['status'] for x in r['rows']]==['native_stopped','native_stopped']
    assert [x['steps_completed'] for x in r['rows']]==[1,1] and len(e.commands)==2
    assert r['counts']['public_step']==dict(attempted=2,completed=2)
    assert r['counts']['physics_step']==dict(attempted=28,completed=28)
    assert all(x['missing_checkpoints']==[5,10,15,20] for x in r['rows'])


def test_future_comparison_arrays_cannot_change_commands(monkeypatch):
    def change(p):
        for x in p:x['comparison_states'][1:]=9999;x['comparison_pixels'][:]=[0,255,0]
    r,s,e,p=execute(monkeypatch,change=change)
    assert r['status']=='overlay_probe_processed'
    for index,a in enumerate(e.commands):assert np.array_equal(a,p[index//20]['actions'][index%20])


def test_fixed_int16_green_and_rgb_math_no_uint8_wrap():
    a=np.zeros((224,224,3),np.uint8);b=np.full_like(a,255);a[0,0]=[0,255,0];b[0,1]=[10,250,20]
    c=m.compare(a,b);d=a.astype(np.int16)-b.astype(np.int16)
    assert c['rgb_MAE']==float(abs(d).mean()) and c['rgb_max_absolute_error']==255
    assert c['green_actual']==c['green_archive']==1 and c['green_xor']==2 and c['green_intersection']==0


def test_unauthorized_does_not_read_or_construct(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('unauthorized source read'))
    with pytest.raises(ValueError):m.execute({},MemoryStore(),lambda:pytest.fail('unauthorized factory'),authorization='wrong')


def test_reference_reader_preserves_fixed_lag0_and_fullslice_binding(monkeypatch):
    payloads=[reference_payload(o) for o in m.prior.ORDINALS];archive=dict(episodes=[]);rows=[]
    arrays=[];images={}
    for o,p in zip(m.prior.ORDINALS,payloads):
        off=100+o;archive['episodes'].append(dict(episode_ordinal=o,status='selected',offset=off))
        a=dict(actions=np.concatenate((p['actions'],p['actions'][:1])),states=p['comparison_states'],pixels=p['comparison_pixels']);arrays.append(a)
        cps=[]
        for j,k in enumerate(m.prior.CHECKPOINTS):
            obs=p['references'][j]['receipt'];images[str(m.REPORT.parent/obs['png'])]=p['references'][j]['pixels']
            cps.append(dict(step=k,archive_global_row=off+k,observation=obs,comparison=dict(archive_state=m.prior.array_id(a['states'][k]),archive_pixels=m.prior.array_id(a['pixels'][j]))))
        rows.append(dict(episode_ordinal=o,lag=0,status='horizon_completed',steps_completed=20,checkpoints=cps))
    monkeypatch.setattr(m.prior,'read_archive',lambda:(deepcopy(archive),deepcopy(arrays)))
    monkeypatch.setattr(m,'old_png',lambda path,h:images[str(path)].copy())
    p,b=m.read_payloads(archive,dict(rows=rows));assert len(p)==2 and len(b)==10
    assert all(x['actions'].shape==(20,2) for x in p)
    archive['extra']='changed'
    # Return a different selected-slice identity, even if image values match.
    monkeypatch.setattr(m.prior,'read_archive',lambda:({},arrays))
    with pytest.raises(ValueError):m.read_payloads(archive,dict(rows=rows))


def test_consumer_checkpoint_exact_field_contract(monkeypatch):
    r,s,e,_=execute(monkeypatch)
    fields=set('step overlay_index with_target png png_sha256 pixels pre post comparison old_off_match off_replay_match'.split())
    for row in r['rows']:
        for f in row['frames']:
            assert set(f)==fields and set(f['pre'])==set('snapshot goal_state goal_pose goal_rgb renderer'.split())
            assert set(f['pixels'])=={'shape','dtype','sha256'} and f['pixels']['dtype']=='|u1'
            assert hashlib.sha256(s.files[f['png']]).hexdigest()==f['png_sha256']
            with Image.open(BytesIO(s.files[f['png']])) as im:
                assert m.prior.array_id(np.asarray(im))==f['pixels']
    assert s.events[-1]['kind']=='run_finished' and r['journal_chain_tail']==s.events[-1]['event_sha256']


def test_actual_action_mismatch_journaled_before_rejection(monkeypatch):
    class WrongAction(Env):
        def step(self,c):
            result=super().step(c);self.latest_action=self.latest_action.astype(float);return result
    r,s,e,_=execute(monkeypatch,WrongAction())
    assert r['status']=='engineering_failed' and r['rows'][0]['steps_completed']==1
    returns=[x for x in s.events if x['kind']=='native_return']
    assert len(returns)==1 and returns[0]['actual_raw']['dtype']=='<f8'
    assert 'actual action' in r['first_error']['message'] and r['rows'][1]['status']=='not_started'


def test_meter_cap_blocks_dispatch_before_callback():
    s=MemoryStore();meter=m.prior.Meter(dict(wall_seconds=100,caps={'explicit_render':1}),s);calls=[]
    meter.call('explicit_render',lambda:calls.append(1))
    with pytest.raises(ValueError):meter.call('explicit_render',lambda:calls.append(2))
    assert calls==[1] and meter.counts['explicit_render']==dict(attempted=1,completed=1)


def test_build_and_verify_source_seal_ram_only(monkeypatch):
    source=dict(bindings={'/RAM/old.py':'a'*64},original_archive={},source_identity_scope='RAM')
    payload=[{},{}]
    monkeypatch.setattr(m,'inputs',lambda:(deepcopy(source),deepcopy(payload)))
    monkeypatch.setattr(m,'sha',lambda path:'a'*64)
    p=m.build();assert m.verify(p)==payload
    p['contract']['lag']=1
    with pytest.raises(ValueError):m.verify(p)
    p=m.build();p['bindings']['/RAM/old.py']='b'*64
    with pytest.raises(ValueError):m.verify(p)
