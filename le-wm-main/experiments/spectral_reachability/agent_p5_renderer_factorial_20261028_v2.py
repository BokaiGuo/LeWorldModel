"""MAIN-only static native target-color diagnostic; zero physics advancement.

Constructor -> original _setup -> public render-view hydration -> reindex.
NOT a contact-cache/dynamical restore. All five original RGBs gate all toggles.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import numpy as np
from experiments.spectral_reachability import agent_p5_renderer_factorial_20261028_v1 as parent

TAG = 'agent_p5_renderer_factorial_20261028_v2'
ROOT, STATES = parent.ROOT, parent.STATES
require, sha, read_json, write_new = parent.require, parent.sha, parent.read_json, parent.write_new
digest = parent.saved.impl.digest
PARENT_SHA = '928c6f2baf0cb877f2878830c3ab62b3003d84daa3a425fe8118f23ca703d6ca'
CAP, FLOOR, WALL = 32*1024**2, 200*1024**2, 600
EXPECTED = dict(env_construct=5, seed=10, setup=5, space_construct=5, body_construct=30,
                view_restore=5, reindex=15, native_render=15, goal_body=15, debug_draw=15, close=5,
                reset=0, set_state=0, control_step=0, space_step=0, native_get_obs=0)
DELTAS = {
    'construct': dict(env_construct=1, seed=1), 'seed': dict(seed=1),
    'setup': dict(setup=1, space_construct=1, body_construct=3),
    'restore': dict(view_restore=1, reindex=3), 'close': dict(close=1),
    **{k: dict(native_render=1, goal_body=1, body_construct=1, debug_draw=1)
       for k in ('baseline', 'visible', 'white')},
}


def protocol():
    return dict(schema=TAG, states=list(STATES), agent='visible only; hidden factor rejected',
        target_order=['visible', 'white'], denominator=10, original_baselines=5, new_native_renders=15,
        original_target='LightGreen, source-inferred with_target=True, subject to exact baseline RGB gate',
        order='all five constructor/seed/setup/hydrate/baseline gates, then all ten toggles, finally close all',
        hydrate='public render-view properties only; original _setup geometry + public property setters/reindex',
        source_state_exception=dict(path='space.current_time_step', saved=0.01, constructed=0.0,
            reason='original read-only most-recent-step getter; never overwrite or fake'),
        complete_dynamic_restore=False, private_contact_cache_restored=False, dynamics_restart_allowed=False,
        target='goal_color only; with_target remains True; finally restore original color object',
        no_pixel_edits=True, no_custom_renderer=True, no_snapshot_set_state=True,
        expected_call_entries=EXPECTED, native_counts='Python entry/return observations; operations have independent durable intent/return',
        neural=0, Space_step=0, reset=0, scientific_admission=False, task_success=None,
        planner_rescoring=False, fresh=False, prospective=False, winner=False,
        object_goal_pose=parent.GOAL, goal_agent_nuisance=parent.AGENT_NUISANCE,
        goal_agent_is_coverage_target=False, wall_seconds=WALL, output_cap_bytes=CAP, free_floor_bytes=FLOOR)


def prepare(manifest, report, receipt):
    require(sha(parent.__file__) == PARENT_SHA, 'frozen parent source')
    old = parent.prepare(manifest, report, receipt)
    records = deepcopy(old['states'])
    for rec in records:
        state = read_json(rec['native_path'], rec['native_sha'])
        require(state['env']['settings']['with_target'] is True, 'actual original target setting')
        require(state['space']['current_time_step'] == 0.01, 'sealed most recent physics dt')
        require(all(b['is_sleeping'] is False for b in state['bodies'].values()), 'no sleeping-body restart')
        rec.update(original_target='visible', original_target_evidence=dict(
            with_target=True, assignment_source=str(ROOT/parent.ENV), assignment_lines=[704,705],
            serialized_goal_color=False, baseline_required=True), source_state=state)
        # reset_to_state is not consumed: no reset is called. All native renderer
        # settings are the exact saved settings; original local shape comes from _setup.
        rec['constructor'] = {**deepcopy(state['env']['settings']), 'reset_to_state': None}
    bindings = dict(old['bindings'])
    bindings[str(Path(parent.__file__).resolve())] = PARENT_SHA
    for p in (Path(__file__), ROOT/'tests'/('test_'+TAG+'.py'), ROOT/'innerdocs'/(TAG+'.md')):
        bindings[str(p.resolve())] = sha(p)
    # Reuse the already sealed native runtime, including package probe, not a new dependency install.
    runtime = ROOT/'results/p5_jepa_native_physics_runtime_20261027_v1/manifest_frozen.json'
    r = read_json(runtime)
    bindings[str(runtime)] = sha(runtime)
    bindings.update(r['seals'])
    for relative,h in {
        '.jepa-physics-site/pymunk/body.py':'3919431decb401d879e686a2506a61378a4ce0b101848628613110e2a7b59d84',
        '.jepa-physics-site/pymunk/_chipmunk.abi3.so':'1224ec1d02916a7001d58e5a7c558f79a7941d3b6132cdfbe58b24665daf3bce',
    }.items(): bindings[str(ROOT/relative)]=h
    require(all(sha(p) == h for p,h in bindings.items()), 'prepare source closure')
    return dict(schema=TAG, status='prepared_static_target_views_NOT_dynamic_restore', protocol=protocol(),
        states=records, bindings=bindings, authority=old['authority'], runtime_manifest=str(runtime),
        source_correction='v1 original_target=white was wrong; bound native setup assigns LightGreen',
        rejected_agent_factor=old['source_contract'],
        supported_design=[dict(state_id=s,agent='visible',target=t) for s in STATES for t in ('visible','white')],
        physical_release_scope='MAIN may run static renderer only after code review; baseline gates may fail',
        scientific_admission=False, dynamics_restart_allowed=False)


def verify(m):
    require(all(sha(p) == h for p,h in m['bindings'].items()), 'binding drift')
    require(m == prepare(**m['authority']), 'full static rendering plan recomputation')


def decode(x):
    if isinstance(x, dict):
        if set(x) == {'nonfinite'}:
            require(x['nonfinite'] in ('+inf','-inf'), 'declared infinity')
            return float(x['nonfinite'])
        if set(x) == {'dtype','shape','values'}:
            a = np.array(x['values'], dtype=x['dtype'])
            require(list(a.shape) == x['shape'] and a.dtype.kind in 'biuf', 'numeric saved dtype/shape')
            return a
        return {k:decode(v) for k,v in x.items()}
    if isinstance(x,list): return [decode(v) for v in x]
    return x


def view_state(source):
    s = deepcopy(source)
    require(s['space']['current_time_step'] == 0.01, 'source dt is fixed')
    s['space']['current_time_step'] = 0.0
    return s


def hydrate_view(env, source, base):
    """No reset/_set_state/physics: render-only public property reconstruction.

    Native shapes must already exactly match the sealed shape/material inventory.
    We do not modify topology or assume a guessed polygon. Full public projection
    equality fails closed for any unsupported property or float roundtrip mismatch.
    """
    require(base.public_state(env)['shapes'] == source['shapes'], 'original _setup local shape/material identity')
    require(base.numeric_json(env.goal_pose) == source['env']['goal_pose'], 'original unchanged native goal')
    for k,v in source['space'].items():
        if k != 'current_time_step': setattr(env.space,k,decode(v))
    for role, body in (('agent',env.agent),('block',env.block),('static',env.space.static_body)):
        s = source['bodies'][role]
        require(body.body_type == s['body_type'] and not s['is_sleeping'], 'native body types/awake state')
        if body.body_type == 0:
            body.mass, body.moment = decode(s['mass']), decode(s['moment'])
        else:
            require(base.numeric_json(body.mass) == s['mass'] and base.numeric_json(body.moment) == s['moment'],
                    'non-dynamic body mass/moment identity')
        # Set CoG and rotation before world position, matching the bound nonlegacy native ordering.
        for field in ('center_of_gravity','angle','position','velocity','angular_velocity','force','torque'):
            setattr(body,field,decode(s[field]))
        env.space.reindex_shapes_for_body(body)  # Public cache update; not a solver step.
    state = source['env']
    env.latest_action = decode(state['latest_action'])
    env.coverage_arr = deepcopy(state['coverage_arr'])
    env.n_contact_points = state['n_contact_points']
    env.teleop = state['teleop']
    rng = source['rng']
    require(env._seed == state['seed'] == rng['fixed_seed'], 'explicit original seed')
    env.np_random.bit_generator.state = decode(rng['generator'])
    env.random_state.set_state(tuple(decode(rng['random_state'])))
    actual = base.public_state(env)
    require(actual == view_state(source), 'all public render-view fields exact, except declared read-only dt')
    return dict(source_full_public_equal=False, render_view_equal=True,
        readonly_difference=dict(path='space.current_time_step',source=source['space']['current_time_step'],actual=actual['space']['current_time_step']),
        shape_mode='exact original native _setup inventory; no shape edits', bodies_restored=3,
        reindex_calls=3, before_render=actual, dynamic_restart_allowed=False)


class Calls:
    def __init__(self, cls, pymunk):
        fns = dict(env_construct=cls.__init__, seed=cls.seed, setup=cls._setup,
            space_construct=pymunk.Space.__init__, body_construct=pymunk.Body.__init__,
            view_restore=hydrate_view, reindex=pymunk.Space.reindex_shapes_for_body,
            native_render=cls._render_frame, goal_body=cls._get_goal_pose_body,
            debug_draw=pymunk.Space.debug_draw, close=cls.close, reset=cls.reset,
            set_state=cls._set_state, control_step=cls.step, space_step=pymunk.Space.step,
            native_get_obs=cls._get_obs)
        require(all(hasattr(f,'__code__') for f in fns.values()), 'native Python entry observation required')
        self.codes = {f.__code__:k for k,f in fns.items()}
        require(len(self.codes) == len(fns), 'unique observed entry codes')
        self.entries = dict.fromkeys(EXPECTED,0); self.returns = dict.fromkeys(EXPECTED,0)
        self.pending = {}; self.errors = []; self.started = False

    def profile(self, frame, event, arg):
        key = self.codes.get(frame.f_code)
        if key is None: return
        if event == 'call':
            self.entries[key] += 1
            if self.entries[key] > EXPECTED[key]:
                self.errors.append('forbidden/overcap entry '+key)
                raise ValueError(self.errors[-1])  # Before native body; failed entry is not refunded.
            self.pending[id(frame)] = key
        elif event == 'return':
            self.returns[key] += 1; self.pending.pop(id(frame),None)

    def start(self):
        require(sys.getprofile() is None, 'do not replace existing profiler')
        sys.setprofile(self.profile); self.started = True

    def stop(self):
        if self.started: sys.setprofile(None); self.started = False

    def receipt(self):
        return dict(entries=dict(self.entries), return_events=dict(self.returns), pending=len(self.pending), errors=list(self.errors))


class Writer:
    def __init__(self,out):
        self.out = Path(out).absolute()
        require(not self.out.exists() and self.out.parent.is_dir() and
            not any(p.is_symlink() for p in (self.out,*self.out.parents)), 'exclusive nonsymlink output')
        self.out.mkdir(); self.started = time.monotonic(); self.bytes = 0; self.seq = 0; self.previous = None
        self.file = (self.out/'journal.jsonl').open('xb')

    def guard(self,reserve=0):
        require(time.monotonic()-self.started < WALL, '600s wall cap')
        require(self.bytes+reserve+1024**2 <= CAP, '32MiB cap plus 1MiB final reserve')
        require(shutil.disk_usage(self.out).free >= FLOOR+reserve+1024**2, '200MiB disk floor')

    def sync(self):
        fd = os.open(self.out,os.O_DIRECTORY|os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)

    def emit(self,name,value,array=False,final=False):
        require('/' not in name, 'local output basename')
        reserve = value.nbytes+1024 if array else len(json.dumps(value,allow_nan=False).encode())*2+1024
        if not final: self.guard(reserve)
        else:
            require(self.bytes+reserve<=CAP and shutil.disk_usage(self.out).free>=FLOOR+reserve, 'terminal reserve/floor')
        path = self.out/name
        if array:
            with path.open('xb') as f: np.save(f,value,allow_pickle=False); f.flush(); os.fsync(f.fileno())
        else: write_new(path,value)
        self.bytes += path.stat().st_size; self.sync()
        return dict(path=name,sha256=sha(path),bytes=path.stat().st_size)

    def event(self,event,final=False):
        require(self.seq<128, 'event cap')
        row = dict(sequence=self.seq,previous_sha256=self.previous,event=event)
        row['entry_sha256'] = digest(row)
        raw = parent.saved.impl.json.dumps(row,sort_keys=True,allow_nan=False).encode()+b'\n'
        if not final: self.guard(len(raw))
        else: require(self.bytes+len(raw)<CAP, 'terminal event cap')
        self.file.write(raw); self.file.flush(); os.fsync(self.file.fileno())
        self.bytes += len(raw); self.seq += 1; self.previous = row['entry_sha256']


def operations():
    return ([(s,k) for s in STATES for k in ('construct','seed','setup','restore','baseline')]+
            [(s,k) for s in STATES for k in ('visible','white')]+[(s,'close') for s in STATES])


def execute(m,manifest_sha,out,loader,stable):
    """Actual control flow; loader injection is RAM-test seam, never CLI input."""
    writer = Writer(out)
    rows = [dict(state_id=s,status='not_run',error=None,shots=[],close_attempted=False,close_returned=False,
                 color_restored=None) for s in STATES]
    cells = [dict(state_id=s,agent='visible',target=t,status='not_run') for s in STATES for t in ('visible','white')]
    envs = {}; calls = None; error = None; all_baselines = False
    def op(s,k,fn,final=False):
        if not final: writer.guard()
        if final and calls.started and sys.getprofile() is None:
            require(calls.errors, 'unexpected profiler loss')
            calls.start()  # Python disables a raising profiler; observe cleanup, do not retry failed work.
        writer.event(dict(phase='intent',state_id=s,operation=k,meter=calls.receipt()),final)
        result = fn()
        writer.event(dict(phase='returned',state_id=s,operation=k,meter=calls.receipt()),final)
        return result
    try:
        stable()
        cls,pymunk,pygame,base,probe = loader()
        writer.emit('runtime_probe.json',probe)
        calls = Calls(cls,pymunk); calls.start()
        def shot(rec,row,target):
            env = envs[row['state_id']]; expected = view_state(rec['source_state'])
            before = base.public_state(env)
            require(before == expected, 'complete before-render view equality')
            color = list(env.goal_color)
            cell = next((c for c in cells if c['state_id']==row['state_id'] and c['target']==target),None)
            if cell is not None: cell['status']='attempted'
            image = op(row['state_id'],target,lambda:env._render_frame('rgb_array'))
            require(image.shape == (224,224,3) and image.dtype == np.uint8, 'native RGB ABI')
            after = base.public_state(env)
            prefix = row['state_id']+'_'+target
            files = dict(rgb=writer.emit(prefix+'.npy',image,array=True),
                before=writer.emit(prefix+'_before.json',before),after=writer.emit(prefix+'_after.json',after))
            record = dict(target=target,files=files,goal_color=color,state_unchanged=before==after)
            row['shots'].append(record)
            require(before == after == expected, 'complete render state/RNG/settings invariant')
            if target == 'baseline':
                require((writer.out/files['rgb']['path']).read_bytes() == Path(rec['rgb']['path']).read_bytes(),
                        'original baseline NPY/RGB byte gate')
            if cell is not None: cell['status']='completed'
            return record
        for rec,row in zip(m['states'],rows):
            s = row['state_id']; row['status'] = 'constructing'
            def construct():
                env = cls(**rec['constructor']); envs[s] = env; return env
            env = op(s,'construct',construct)
            op(s,'seed',lambda:env.seed(rec['source_state']['env']['seed']))
            op(s,'setup',env._setup)
            require(env.with_target is True and list(env.goal_color)==list(pygame.Color('LightGreen')), 'original setup target color source')
            row['restore'] = op(s,'restore',lambda:hydrate_view(env,rec['source_state'],base))
            writer.emit(s+'_restore.json',row['restore'])
            shot(rec,row,'baseline')
            row['status'] = 'baseline_verified'
        require(all(r['status']=='baseline_verified' for r in rows), 'all-five baseline gate before new factor views')
        all_baselines = True
        for rec,row in zip(m['states'],rows):
            env = envs[row['state_id']]; original = env.goal_color
            try:
                for target,name in (('visible','LightGreen'),('white','White')):
                    env.goal_color = pygame.Color(name)
                    shot(rec,row,target)
            finally:
                env.goal_color = original
                row['color_restored'] = env.goal_color is original
                require(row['color_restored'] and base.public_state(env)==view_state(rec['source_state']), 'finally color and complete state restoration')
            row['status'] = 'completed'
        preclose={k:v-(5 if k=='close' else 0) for k,v in EXPECTED.items()}
        require(calls.receipt()==dict(entries=preclose,return_events=preclose,pending=0,errors=[]), 'preclose native census')
    except Exception as exc:
        error = type(exc).__name__+': '+str(exc)
        for row in rows:
            if row['status']!='completed': row['error']=error
    finally:
        for row in rows:
            if row['state_id'] in envs:
                try:
                    row['close_attempted'] = True
                    op(row['state_id'],'close',envs[row['state_id']].close,True)
                    row['close_returned'] = True
                except Exception as exc:
                    error = error or ('close: '+str(exc))
        if calls is not None: calls.stop()
        try: stable(); writer.guard()
        except Exception as exc: error = error or ('final verification/resources: '+str(exc))
    meter = None if calls is None else calls.receipt()
    if error is None and meter != dict(entries=EXPECTED,return_events=EXPECTED,pending=0,errors=[]):
        error = 'native complete entry/return census mismatch'
    result = dict(schema=TAG+'.report',status='all10_static_target_views_recorded' if error is None else 'static_render_failed',
        manifest_sha256=manifest_sha,manifest_digest=digest(m),protocol=protocol(),rows=rows,cells=cells,global_error=error,
        all_five_baselines_before_toggles=all_baselines,meter=meter,profiler_released=not bool(calls and calls.started),
        wall_seconds=time.monotonic()-writer.started,scientific_admission=False,neural_executed=False,
        dynamics_restart_allowed=False,complete_dynamic_restore=False,Space_step_executed=False)
    writer.event(dict(phase='terminal',summary_sha256=digest(result)),True)
    writer.file.close()
    result['artifacts'] = {p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in writer.out.iterdir()}
    writer.emit('report.json',result,final=True)
    return result


def load_runtime(m):
    require(os.environ.get('PYTHONPATH')==str(ROOT/'.jepa-physics-site') and os.environ.get('PYTHONNOUSERSITE')=='1', 'isolated original physics overlay required')
    require(not any(n=='torch' or n.startswith('torch.') for n in sys.modules), 'no neural-loaded native process')
    name = 'experiments.spectral_reachability.p5_jepa_factual_pilot_20261027_v1'
    import importlib
    base = importlib.import_module(name)
    require(sha(base.__file__)==parent.PIN[parent.PILOT], 'sealed public-state reader')
    runtime,old = base.load_runtime(ROOT)
    probe = runtime.probe(ROOT,old,Path(m['runtime_manifest']))
    require(probe['status']=='native_import_ready_physics_not_executed', 'original native runtime probe')
    path = ROOT/parent.ENV
    spec = importlib.util.spec_from_file_location(TAG+'_native',path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    import pymunk, pygame
    require(Path(module.PushTEnv.__init__.__code__.co_filename).resolve()==path and
        Path(pymunk.__file__).resolve().is_relative_to(ROOT/'.jepa-physics-site') and
        Path(pygame.__file__).resolve().is_relative_to(ROOT/'.jepa-physics-site'), 'original environment and overlay provenance')
    require(not any(n=='torch' or n.startswith('torch.') for n in sys.modules), 'no neural imports')
    return module.PushTEnv,pymunk,pygame,base,probe


def run(manifest,manifest_sha,out,authorized=False):
    require(authorized, 'MAIN authorization required')
    m = read_json(manifest,manifest_sha); verify(m)
    out = Path(out).absolute()
    require(out.parent==ROOT/'results' and out.name.startswith(TAG+'_'), 'new exclusive results child')
    require(signal.getsignal(signal.SIGALRM)==signal.SIG_DFL and signal.getitimer(signal.ITIMER_REAL)==(0.,0.), 'no preexisting timer')
    def timeout(*args): raise TimeoutError('600s static render cap')
    signal.signal(signal.SIGALRM,timeout); signal.alarm(WALL)
    try:
        return execute(m,manifest_sha,out,lambda:load_runtime(m),lambda:verify(m))
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM,signal.SIG_DFL)


def consume(m,manifest_sha,report,report_sha):
    verify(m)
    r = read_json(report,report_sha); out = Path(report).parent
    require(r['schema']==TAG+'.report' and r['manifest_sha256']==manifest_sha and r['manifest_digest']==digest(m)
        and r['protocol']==protocol() and r['status']=='all10_static_target_views_recorded' and r['global_error'] is None, 'actual complete report authority')
    require(r['meter']==dict(entries=EXPECTED,return_events=EXPECTED,pending=0,errors=[]) and r['profiler_released'] is True,
            'complete actual calls and cleanup')
    require(r['scientific_admission'] is r['neural_executed'] is r['dynamics_restart_allowed'] is r['complete_dynamic_restore'] is r['Space_step_executed'] is False,
            'static only, not scientific or dynamic release')
    require(r['all_five_baselines_before_toggles'] is True and 0<=r['wall_seconds']<WALL, 'all-five phase gate/wall')
    require(r['cells']==[dict(state_id=s,agent='visible',target=t,status='completed') for s in STATES for t in ('visible','white')], 'complete ten-cell denominator')
    actual = {p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.name!='report.json'}
    require(actual==r['artifacts'] and sum(x['bytes'] for x in actual.values())+Path(report).stat().st_size<=CAP, 'exact saved inventory/cap')
    require([x['state_id'] for x in r['rows']]==list(STATES), 'all-five state order')
    names = {'runtime_probe.json','journal.jsonl'}
    for source,row in zip(m['states'],r['rows']):
        s = row['state_id']; expected = view_state(source['source_state'])
        require(row['status']=='completed' and row['error'] is None and row['close_attempted'] is row['close_returned'] is row['color_restored'] is True, 'full state completion/close/color restoration')
        restore = read_json(out/(s+'_restore.json'))
        require(restore==row['restore'] and restore==dict(source_full_public_equal=False,render_view_equal=True,
            readonly_difference=dict(path='space.current_time_step',source=0.01,actual=0.0),
            shape_mode='exact original native _setup inventory; no shape edits',bodies_restored=3,reindex_calls=3,
            before_render=expected,dynamic_restart_allowed=False), 'exact reconstruction boundary')
        names.add(s+'_restore.json')
        require([x['target'] for x in row['shots']]==['baseline','visible','white'], 'baseline and both target conditions')
        for shot in row['shots']:
            target = shot['target']; prefix=s+'_'+target
            color = [255,255,255,255] if target=='white' else [144,238,144,255]
            require(shot['goal_color']==color and shot['state_unchanged'] is True, 'actual named native draw color/state')
            wanted_files = dict(rgb=prefix+'.npy',before=prefix+'_before.json',after=prefix+'_after.json')
            for key,name in wanted_files.items():
                require(shot['files'][key]==dict(path=name,**actual[name]), 'exact shot file identity')
                names.add(name)
            require(read_json(out/wanted_files['before'])==read_json(out/wanted_files['after'])==expected, 'complete exact before/after state')
            pixels=np.load(out/wanted_files['rgb'],allow_pickle=False,max_header_size=1024)
            require(pixels.shape==(224,224,3) and pixels.dtype==np.uint8, 'native saved RGB ABI')
            if target=='baseline': require((out/wanted_files['rgb']).read_bytes()==Path(source['rgb']['path']).read_bytes(), 'sealed original baseline byte gate')
        require((out/(s+'_visible.npy')).read_bytes()==(out/(s+'_baseline.npy')).read_bytes(), 'unchanged native visible baseline repeat')
    require(set(actual)==names and read_json(out/'runtime_probe.json')['status']=='native_import_ready_physics_not_executed', 'exact expected producer layout/probe')
    require((out/'journal.jsonl').stat().st_size<=4*1024**2, 'bounded journal')
    events = [json.loads(line) for line in (out/'journal.jsonl').read_text().splitlines()]
    require(len(events)==81, '40 fixed operation intent/returns plus terminal')
    prev=None; counts=dict.fromkeys(EXPECTED,0)
    for i,e in enumerate(events):
        require(e['sequence']==i and e['previous_sha256']==prev and e['entry_sha256']==digest({k:v for k,v in e.items() if k!='entry_sha256'}), 'journal chain')
        prev=e['entry_sha256']
        if i==80:
            core={k:v for k,v in r.items() if k!='artifacts'}
            require(e['event']==dict(phase='terminal',summary_sha256=digest(core)), 'terminal report identity')
            continue
        s,k=operations()[i//2]; val=e['event']
        require(val['state_id']==s and val['operation']==k and val['phase']==('intent' if i%2==0 else 'returned'), 'fixed global phase order')
        if i%2:
            for key,n in DELTAS[k].items(): counts[key]+=n
        require(val['meter']==dict(entries=counts,return_events=counts,pending=0,errors=[]), 'actual per-operation census')
    verify(m); require(sha(report)==report_sha, 'final report drift')
    return dict(schema=TAG+'.consume',status='all10_saved_static_target_views_verified',report_sha256=report_sha,
        manifest_sha256=manifest_sha,denominator=10,baselines=5,new_native_renders=15,Space_step=0,
        complete_dynamic_restore=False,readonly_difference=protocol()['source_state_exception'],
        agent_hidden_released=False,scientific_admission=False,task_success=None,dynamics_restart_allowed=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','verify','run','consume')); p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--manifest-sha256'); p.add_argument('--authority',nargs=3,type=Path)
    p.add_argument('--output',type=Path); p.add_argument('--report',type=Path); p.add_argument('--report-sha256')
    p.add_argument('--main-authorized',action='store_true'); a=p.parse_args(argv)
    if a.command=='prepare':
        require(a.authority is not None,'actual v5 prepare/report/receipt required')
        result=prepare(*a.authority); write_new(a.manifest,result)
    else:
        require(a.manifest_sha256 and sha(a.manifest)==a.manifest_sha256,'actual manifest SHA')
        m=read_json(a.manifest,a.manifest_sha256)
        if a.command=='verify': verify(m); result=dict(status='static_render_plan_verified')
        elif a.command=='run':
            require(a.output is not None,'exclusive run output'); result=run(a.manifest,a.manifest_sha256,a.output,a.main_authorized)
        else:
            require(a.report and a.report_sha256 and a.output,'actual report SHA and exclusive receipt output')
            result=consume(m,a.manifest_sha256,a.report,a.report_sha256); write_new(a.output,result)
    print(json.dumps({'status':result['status'],'scientific_admission':False}))
    return 2 if result['status']=='static_render_failed' else 0


if __name__=='__main__': raise SystemExit(main())
