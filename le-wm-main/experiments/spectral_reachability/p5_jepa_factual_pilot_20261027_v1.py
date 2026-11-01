"""Prepare-only by default; main-only fresh native zero-action factual pilot.

No physics import at module import/prepare. No neural dependency or pickle.
Run requires explicit confirmation, frozen SHA and reviewed target runtime.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

NAME='p5_jepa_factual_pilot_20261027_v1'
RUNTIME='p5_jepa_native_physics_runtime_20261027_v1'
RUNTIME_SHA='76953de72838fcce618862f175f4242a18b3030ec11c6342196f88d919df48b4'
ROOT=Path(__file__).resolve().parents[2]
SEED=2026102701
ROLES=('a','b')
CONTROLS=15
BODY_FIELDS=('position','velocity','angle','angular_velocity','force','torque','center_of_gravity','mass','moment','body_type','is_sleeping')
SPACE_FIELDS=('gravity','damping','iterations','collision_slop','collision_bias','collision_persistence','idle_speed_threshold','sleep_time_threshold','current_time_step')
CTOR=dict(legacy=False,block_cog=None,damping=None,render_action=False,render_size=224,
    reset_to_state=None,relative=True,action_scale=100,with_velocity=True,with_target=True,shape='T',color='LightSlateGray')


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha(path):
    p=Path(path)
    if p.is_symlink() or not p.is_file():raise ValueError('regular non-symlink file required: '+str(p))
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def numeric_json(value):
    """Lossless float64 JSON numbers/explicit infinities; no NaN/pickle/repr."""
    if isinstance(value,np.ndarray):
        if value.dtype.kind not in 'biuf':raise ValueError('numeric ndarray required')
        return dict(dtype=value.dtype.str,shape=list(value.shape),values=numeric_json(value.tolist()))
    if isinstance(value,np.generic):return numeric_json(value.item())
    if value is None or isinstance(value,(str,bool,int)):return value
    if isinstance(value,float):
        if np.isnan(value):raise ValueError('NaN not allowed')
        if np.isinf(value):return {'nonfinite':'+inf' if value>0 else '-inf'}
        return value
    if isinstance(value,dict):
        if any(not isinstance(k,str) for k in value):raise ValueError('string JSON keys required')
        return {k:numeric_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [numeric_json(x) for x in value]
    # Native Vec2d and pygame Color expose iterable numeric components.
    if type(value).__name__ in ('Vec2d','Color'):return [numeric_json(x) for x in value]
    raise ValueError('unreviewed public value type: '+type(value).__name__)


def rng_state(env):
    return numeric_json(dict(generator=env.np_random.bit_generator.state,random_state=env.random_state.get_state(),fixed_seed=env._seed))


def public_state(env):
    bodies={'agent':env.agent,'block':env.block,'static':env.space.static_body}
    if any(all(b is not known for known in bodies.values()) for b in env.space.bodies):
        raise ValueError('unmapped native body')
    body_records={role:{k:numeric_json(getattr(b,k)) for k in BODY_FIELDS} for role,b in bodies.items()}
    shapes=[]
    for s in env.space.shapes:
        role=next((k for k,b in bodies.items() if s.body is b),None)
        if role is None:raise ValueError('unmapped shape body')
        name=type(s).__name__
        item=dict(body=role,type=name,radius=float(s.radius),
            friction=float(s.friction),elasticity=float(s.elasticity),surface_velocity=numeric_json(s.surface_velocity),
            collision_type=int(s.collision_type),sensor=bool(s.sensor),color=numeric_json(s.color),
            filter={k:int(getattr(s.filter,k)) for k in ('group','categories','mask')})
        if name=='Circle':item['offset']=numeric_json(s.offset)
        elif name=='Segment':item.update(a=numeric_json(s.a),b=numeric_json(s.b))
        elif name=='Poly':item['vertices']=[numeric_json(v) for v in s.get_vertices()]
        else:raise ValueError('unreviewed shape: '+name)
        shapes.append(item)
    # Native sets have address-dependent order; canonicalize semantic geometry,
    # retain duplicates. No pointer/arbiter id is treated as scientific state.
    shapes.sort(key=canonical)
    return dict(bodies=body_records,shapes=shapes,
        space={k:numeric_json(getattr(env.space,k)) for k in SPACE_FIELDS},rng=rng_state(env),
        env=numeric_json(dict(seed=env._seed,goal_pose=env.goal_pose,latest_action=env.latest_action,
            n_contact_points=env.n_contact_points,coverage_arr=env.coverage_arr,
            settings={k:getattr(env,k) for k in CTOR if k!='reset_to_state'},
            sim_hz=env.sim_hz,control_hz=env.control_hz,k_p=env.k_p,k_v=env.k_v,
            success_threshold=env.success_threshold,teleop=env.teleop)))


def sync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def mkdir(path):
    Path(path).mkdir(exist_ok=False)
    sync_dir(Path(path).parent)


def save_json(path,value):
    with Path(path).open('xb') as f:
        f.write(canonical(value)+b'\n');f.flush();os.fsync(f.fileno())
    sync_dir(Path(path).parent)


def save_array(path,value):
    a=np.asarray(value)
    if a.dtype.kind not in 'biuf':raise ValueError('no object/pickle arrays')
    with Path(path).open('xb') as f:
        np.save(f,a,allow_pickle=False);f.flush();os.fsync(f.fileno())
    sync_dir(Path(path).parent)


class Journal:
    def __init__(self,path):
        self.path=Path(path);self.sequence=0;self.previous=None
        self.file=self.path.open('xb');self.file.flush();os.fsync(self.file.fileno());sync_dir(self.path.parent)

    def append(self,event):
        row=dict(sequence=self.sequence,previous_sha256=self.previous,event=event)
        digest=hashlib.sha256(canonical(row)).hexdigest()
        row['entry_sha256']=digest
        self.file.write(canonical(row)+b'\n');self.file.flush();os.fsync(self.file.fileno())
        self.sequence+=1;self.previous=digest
        return digest

    def close(self):self.file.close()


class NativeCalls:
    """Python call-profile observation, not a monkeypatch of native methods."""
    def __init__(self,cls,pymunk):
        self.codes={};self.counts=Counter();self.space_steps=[];self.context=None;self.started=False
        targets={'env_construct':cls.__init__,'seed':cls.seed,'reset':cls.reset,'control_step':cls.step,
            'native_get_obs':cls._get_obs,'native_render':cls._render_frame,'setup':cls._setup,
            'set_state':cls._set_state,'native_goal_body':cls._get_goal_pose_body,'close':cls.close,
            'space_construct':pymunk.Space.__init__,'space_step':pymunk.Space.step,'body_construct':pymunk.Body.__init__}
        for name,fn in targets.items():
            if not hasattr(fn,'__code__'):raise ValueError('native profiling interface unavailable: '+name)
            self.codes[fn.__code__]=name

    def profile(self,frame,event,arg):
        if event!='call' or frame.f_code not in self.codes:return
        name=self.codes[frame.f_code];self.counts[name]+=1
        if name=='space_step':self.space_steps.append(dict(context=self.context,dt=float(frame.f_locals['dt'])))

    def start(self):
        if sys.getprofile() is not None:raise ValueError('existing profiler; refuse to replace it')
        sys.setprofile(self.profile);self.started=True

    def stop(self):
        if self.started:
            sys.setprofile(None);self.started=False

    def receipt(self):return dict(python_call_entries=dict(self.counts),space_step_events=self.space_steps.copy())


def operation(journal,calls,role,op,details,fn):
    calls.context=dict(role=role,operation=op,frame=details.get('frame'))
    journal.append(dict(phase='intent',role=role,operation=op,details=details,counters_before=calls.receipt()))
    # append fsync completed before fn; interrupted intents are INDETERMINATE,
    # never auto-resumed or silently retried.
    value=fn()
    journal.append(dict(phase='returned',role=role,operation=op,counters_after=calls.receipt()))
    return value


def capture(env,observation,state,transition,path):
    mkdir(path)
    visual=np.asarray(observation['visual']);state=np.asarray(state);proprio=np.asarray(observation['proprio'])
    # Save native returns before scientific shape/identity gates; no lossy codec.
    for name,a in (('rgb',visual),('state',state),('proprio',proprio)):
        save_array(path/(name+'.npy'),a)
    save_json(path/'transition.json',numeric_json(transition))
    public=public_state(env)
    save_json(path/'full_public_state.json',public)
    if (visual.shape!=(224,224,3) or visual.dtype!=np.uint8 or state.shape!=(7,) or state.dtype!=np.float32
            or proprio.shape!=(4,) or proprio.dtype!=np.float32 or not np.isfinite(state).all()
            or not np.isfinite(proprio).all()):raise ValueError('native observation contract rejected; raw returns retained')
    if not np.array_equal(proprio,state[[0,1,5,6]]):raise ValueError('native proprio/state mismatch')
    return dict(frame=int(path.name.split('_')[-1]),files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(path.iterdir())},
                state=state.tolist(),proprio=proprio.tolist())


def collect_trajectory(cls,role,out,journal,calls):
    """Internal collector; cls injected only by tests or strict main loader."""
    env=None;records=[];branch=out/role;mkdir(branch)
    try:
        env=operation(journal,calls,role,'construct',dict(config=CTOR),lambda:cls(**CTOR))
        save_json(branch/'constructor_rng.json',rng_state(env))
        operation(journal,calls,role,'seed',dict(seed=SEED),lambda:env.seed(SEED))
        save_json(branch/'pre_reset_rng.json',rng_state(env))
        obs,state=operation(journal,calls,role,'reset',dict(seed=SEED,expected_space_steps=1,dt=.01),env.reset)
        record=capture(env,obs,state,dict(kind='reset'),branch/'frame_00');records.append(record)
        journal.append(dict(phase='assets_durable',role=role,record=record))
        for frame in range(1,CONTROLS+1):
            u=np.zeros(2,dtype=np.float64)
            agent=np.array(env.agent.position,dtype=np.float64)
            command=dict(frame=frame,raw_env_argument=u.tolist(),pixel_displacement=(100*u).tolist(),
                agent_xy_before=agent.tolist(),absolute_target=(agent+100*u).tolist(),
                expected_substeps=10,dt=.01,pre_public_sha256=record['files']['full_public_state.json']['sha256'])
            save_json(branch/f'command_{frame:02d}.json',command)
            obs,reward,done,info=operation(journal,calls,role,'step',command,lambda:env.step(u.copy()))
            # step returns authoritative native float32 state inside info: no
            # extra _get_obs or render calls are used by the collector.
            record=capture(env,obs,info['state'],dict(kind='step',reward=reward,done=done,info=info),branch/f'frame_{frame:02d}')
            records.append(record);journal.append(dict(phase='assets_durable',role=role,record=record))
            if done:raise ValueError('unexpected native done; retain interrupted trajectory, no replacement')
        return records
    finally:
        if env is not None:operation(journal,calls,role,'close',{},env.close)


def compare_records(a,b):
    if len(a)!=16 or len(b)!=16:raise ValueError('two complete16-frame trajectories required')
    rows=[]
    for frame,(left,right) in enumerate(zip(a,b)):
        if left['frame']!=frame or right['frame']!=frame:raise ValueError('frame identity/order mismatch')
        equal={name:left['files'][name]['sha256']==right['files'][name]['sha256']
            for name in ('rgb.npy','state.npy','proprio.npy','full_public_state.json','transition.json')}
        rows.append(dict(frame=frame,exact=equal,all_exact=all(equal.values())))
    return rows


def protocol():
    return dict(schema=NAME,seed=SEED,roles=list(ROLES),constructor=CTOR,controls_per_env=15,
        action='raw native env argument u=[0.,0.], float64; z=(u-mean)/std; not normalizedzero, not None',
        root='fresh post-reset native state; reset includes one .01s physics step',
        expected=dict(env_construct=2,seed=4,reset=2,control_step=30,space_step=302,native_render=32,native_get_obs=32,close=2),
        requested_native_rgb_frames=32,simulation_seconds_per_env=1.51,
        comparison='all16frames state/proprio/RGB/full-public-state/transition file SHA exact; no tolerances or sample replacement',
        journal='fsync hash-chained intent BEFORE each native construct/seed/reset/step/close; command+prestate binding durable before step; no resume after incomplete intent',
        snapshot_boundary='all declared public properties/RNG/config, NOT private contact cache serialization or arbitrary snapshot restore',
        profiler_boundary='sys.setprofile call entries of unmodified native Python functions, dt per Space.step; no wrapping/patching physics or rendering',
        selection='first fixed fresh seed only; collect both full trajectories then compare, retain every mismatch; execution exception fails closed with partial assets and not_run rows',
        neural_execution=False,training=False,action_fidelity_claim=False,control_benefit_claim=False,P5_admission=False)


def runtime_manifest(root):
    path=root/f'results/{RUNTIME}/manifest_frozen.json'
    if sha(path)!=RUNTIME_SHA:raise ValueError('frozen runtime manifest rejected')
    value=json.loads(path.read_text())
    for p,h in value['seals'].items():
        if sha(p)!=h:raise ValueError('runtime source seal drift: '+p)
    if value['fresh_calibration_proposal']['env_constructor']!=CTOR or value['fresh_calibration_proposal']['seeds'][0]!=SEED:
        raise ValueError('runtime fresh-root proposal differs')
    return path,value


def prepare(root):
    path,parent=runtime_manifest(root)
    seals=dict(parent['seals'])
    for p in [path,Path(__file__).resolve(),root/'tests'/f'test_{NAME}.py',root.parent/'docs'/f'{NAME}.md']:
        seals[str(p)]=sha(p)
    return dict(schema=NAME,status='prepared_no_physics',protocol=protocol(),seals=seals,
        runtime_manifest_sha256=RUNTIME_SHA,physics_executed=False,neural_executed=False,P5_admission=False)


def verify(root,manifest):
    expected=prepare(root)
    if manifest!=expected:raise ValueError('pilot frozen protocol/source manifest rejected')


def load_runtime(root):
    _,parent=runtime_manifest(root)
    path=root/f'experiments/spectral_reachability/{RUNTIME}.py'
    if sha(path)!=parent['seals'][str(path)]:raise ValueError('runtime module SHA mismatch')
    spec=importlib.util.spec_from_file_location(NAME+'_bound_runtime',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m,parent


def run(root,manifest,manifest_path,expected_sha,out):
    if sha(manifest_path)!=expected_sha:raise ValueError('explicit manifest SHA rejected')
    verify(root,manifest)
    if out.exists():raise FileExistsError('exclusive new output directory required; no overwrite/resume')
    if out.parent.resolve()!=root/f'results/{NAME}':raise ValueError('output must be a new direct pilot results child')
    mkdir(out);journal=Journal(out/'journal.jsonl');calls=None;collected={};problem=None
    rows=[dict(frame=i,status='not_run') for i in range(16)]
    source_after=False
    try:
        journal.append(dict(phase='preflight_intent',protocol=protocol(),manifest_sha256=expected_sha))
        runtime,parent=load_runtime(root)
        probe=runtime.probe(root,parent,root/f'results/{RUNTIME}/manifest_frozen.json')
        save_json(out/'runtime_probe.json',probe)
        if probe['status']!='native_import_ready_physics_not_executed':raise ValueError('native runtime probe rejected; no env constructed')
        # Second import is explicit; probe only verified class importability.
        # No physics constructor has been called in either import.
        source=root/runtime.ENV_REL
        spec=importlib.util.spec_from_file_location(NAME+'_official_native',source)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        if Path(m.PushTEnv.__init__.__code__.co_filename).resolve()!=source:raise ValueError('native constructor provenance rejected')
        import pymunk  # Main-only path, after exact overlay/source probe.
        calls=NativeCalls(m.PushTEnv,pymunk)
        calls.start()
        try:
            for role in ROLES:collected[role]=collect_trajectory(m.PushTEnv,role,out,journal,calls)
        finally:calls.stop()
        rows=compare_records(collected['a'],collected['b'])
        verify(root,manifest);source_after=True
        counts=calls.receipt()['python_call_entries']
        if any(counts.get(k,0)!=v for k,v in protocol()['expected'].items()):raise ValueError('actual native call counts differ')
        if any(e['dt']!=.01 for e in calls.space_steps):raise ValueError('actual physics timestep differs')
    except Exception as exc:
        problem=type(exc).__name__+': '+str(exc)
    finally:
        if calls is not None and calls.started:calls.stop()
        try:verify(root,manifest);source_after=True
        except Exception as exc:source_after=False;problem=(problem+'; ' if problem else '')+str(exc)
    exact=problem is None and all(row.get('all_exact',False) for row in rows)
    result=dict(schema=NAME,status='fresh_native_zero_pilot_recorded' if exact else 'fresh_native_zero_pilot_failed',
        protocol=protocol(),manifest_file_sha256=expected_sha,error=problem,rows=rows,
        trajectory_frames={k:len(v) for k,v in collected.items()},source_hashes_unchanged=source_after,
        native_call_receipts=calls.receipt() if calls else None,
        physics_executed=bool(calls and calls.counts.get('space_step',0)),neural_executed=False,
        evidence_scope='fresh factual zero-action repeat identity only; no action fidelity/forecast/control/encoder equivalence or P5 admission',P5_admission=False)
    journal.append(dict(phase='terminal',status=result['status'],summary_sha256=hashlib.sha256(canonical(result)).hexdigest()))
    journal.close();result['journal_sha256']=sha(out/'journal.jsonl')
    save_json(out/'report.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','run'],nargs='?',default='prepare')
    p.add_argument('--manifest',type=Path,default=ROOT/f'results/{NAME}/manifest_frozen.json')
    p.add_argument('--manifest-sha256')
    p.add_argument('--confirm-physics',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/f'results/{NAME}/pilot_main_v1')
    args=p.parse_args()
    if args.command=='prepare':value=prepare(ROOT)
    else:
        if not args.confirm_physics or not args.manifest_sha256:p.error('main-only run requires --confirm-physics and --manifest-sha256')
        value=run(ROOT,json.loads(args.manifest.read_text()),args.manifest,args.manifest_sha256,args.output)
    print(json.dumps(value,indent=2,sort_keys=True,allow_nan=False))
    if value['status']=='fresh_native_zero_pilot_failed':raise SystemExit(2)


if __name__=='__main__':main()
