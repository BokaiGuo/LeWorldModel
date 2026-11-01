"""MAIN-only physics task construction; prepare/mock are neural/physics free.

Fixed 2--4 slots, one deterministic witness each, endpoint-derived native goal.
No direct-difficulty requirement, trained component, outcome-selected role or retry.
"""
from __future__ import annotations
import argparse
import base64
from copy import deepcopy
import hashlib
import gzip
import inspect
import io
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import p1_conditional_procedural_source_20261027_v1 as source

NAME = 'p1_constructive_reachable_source_20261027_v1'
ROOT = source.PROJECT
FLOOR, CAP, FINAL = 200*1024**2, 8*1024**2, 128*1024
AUTH = 'MAIN_CONSTRUCTIVE_SOURCE_PHYSICS_ONLY'
ENV = ROOT/'.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py'
ENV_SHA = 'c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2'
require = source.require


def raw(value): return (source.canonical(value)+'\n').encode()
def image_sha(value): return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def image(value):
    a = np.asarray(value); require(a.shape == (224,224,3) and a.dtype == np.uint8, 'native uint8 RGB224'); return a.copy()


def build(root_seeds, reset_seeds, wall_seconds, *, history=None):
    require(len(root_seeds) == len(reset_seeds) and 2 <= len(root_seeds) <= 4, 'fixed2--4 slots')
    require(all(type(x) is int and 0 < x < 2**31 for x in root_seeds+reset_seeds) and len(set(root_seeds+reset_seeds)) == 2*len(root_seeds), 'distinct explicit positive seeds')
    require(type(wall_seconds) is int and 0 < wall_seconds <= 600, 'explicit wall cap <=600s')
    history = source.collect_current_exclusions() if history is None else history
    source.validate_exclusions(history)
    for s in root_seeds+reset_seeds:
        require(all(s != x['seed'] for x in history['seed_families']) and all(not x['span'][0] <= s < x['span'][1] for x in history['reset_spans']), 'seed intersects historical/reserved source')
    n = len(root_seeds); files = [Path(__file__), Path(source.__file__), ENV,
        ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md'),
        ENV.parent.parent/'utils.py', ROOT/'.venv/lib/python3.12/site-packages/stable_worldmodel/spaces.py']
    bindings = {str(f.resolve()):source.file_sha(f) for f in files}
    require(bindings[str(ENV)] == ENV_SHA, 'original native source hash')
    return dict(schema=NAME+'.prepare', slots=[dict(index=i, root_seed=a, reset_seed=b, sign=(-1 if i%2 else 1)) for i,(a,b) in enumerate(zip(root_seeds,reset_seeds))],
        bindings=bindings, exclusions=history, global_freshness='unknown_calibration_only',
        recipe=dict(position_jitter=16., angle_offset_range=[-np.pi,np.pi], edge_fraction_range=[.15,.85], pusher_surface_offset=15.),
        witness='5 repeated radial logit=.5 then5 logit=.25radial+sign*.5tangent; fixed sign by slot; world geometry only in offline builder',
        environment='swm/PushT-v1', with_target=False, warmup=10, horizon=10, replay_episodes=2,
        packet_logit_cap=float(.75*np.sqrt(5)), wall_seconds=wall_seconds, caps=dict(reset=6*n, public_step=60*n,
            explicit_render=63*n, goal_render=6*n, set_state=12*n, physics_step=612*n, goal_bind=2*n),
        output_cap_bytes=CAP, disk_floor_bytes=FLOOR, new_neural_calls=0, training=False, P1_admission=False,
        selection='one fixed witness per slot; no replacement; failed slots retained; native success is reach label',
        goal_attachment='after original10-step observed history, native _set_goal_state only + _goal=actual endpoint RGB; no state setter/physics',
        deployment_payload='only own replay RGB_history/raw_action_history/goal_RGB; certificate actions/states forbidden')


def verify(p):
    require(p['schema'] == NAME+'.prepare' and len(p['slots']) in (2,3,4), 'constructive independent schema')
    for f,h in p['bindings'].items(): require(source.file_sha(f) == h, 'source seal drift: '+f)
    for f,h in p['exclusions']['files'].items():
        # Sealed metadata resolver emits this semantic ID, not a pathname.
        # Its value is bound by the complete prepared exclusions, not file SHA.
        if f == 'metadata_authority_content_id':
            source.hash_value(h); continue
        path = Path(f); path = path if path.is_absolute() else ROOT/path
        require(source.file_sha(path) == h, 'historical exclusion input drift')
    # Reconstruct without refreshing the historical census.
    require(p['horizon'] == p['warmup'] == 10 and p['replay_episodes'] == 2 and p['with_target'] is False and
        p['packet_logit_cap'] == float(.75*np.sqrt(5)) and p['output_cap_bytes'] == CAP and p['disk_floor_bytes'] == FLOOR, 'fixed geometry/budget')
    slots=p['slots']
    require(p == build([s['root_seed'] for s in slots],[s['reset_seed'] for s in slots],p['wall_seconds'],history=p['exclusions']), 'complete prepared recipe/budget reconstruction')
    return p


def prepare(path, **kwargs):
    p = build(**kwargs); value = prepared_bytes(p); path = Path(path)
    require(len(value) < 200_000 and not path.exists() and shutil.disk_usage(path.parent).free >= FLOOR+len(value)+4096, 'exclusive small prepare +reserve')
    with path.open('xb') as f: f.write(value); f.flush(); os.fsync(f.fileno())
    return source.file_sha(path)


def prepared_bytes(p):
    # Lossless metadata-only envelope: never copy large source protocols.
    return raw(dict(schema=NAME+'.envelope',plan_sha256=source.digest(p),
        gzip_base64=base64.b64encode(gzip.compress(raw(p),mtime=0)).decode('ascii')))


def read_prepared(value):
    require(len(value) < 200_000, 'small prepared envelope')
    e=json.loads(value)
    require(set(e) == {'schema','plan_sha256','gzip_base64'} and e['schema'] == NAME+'.envelope', 'exact prepared envelope')
    with gzip.GzipFile(fileobj=io.BytesIO(base64.b64decode(e['gzip_base64'],validate=True))) as f:
        decoded=f.read(2*1024**2+1)
    require(len(decoded) <= 2*1024**2, 'bounded metadata decompression')
    p=json.loads(decoded);require(source.digest(p) == e['plan_sha256'], 'envelope semantic SHA')
    return verify(p)


def witness(state, sign, cap):
    s = source.vector(state); v = s[2:4]-s[:2]; length = np.linalg.norm(v)
    require(length > 0 and sign in (-1,1), 'finite radial recipe')
    radial = v/length; tangent = np.array([-radial[1],radial[0]])
    logits = np.concatenate((np.tile(.5*radial,(5,1)),np.tile(.25*radial+sign*.5*tangent,(5,1)))).astype(np.float32)
    require((np.linalg.norm(logits.reshape(2,10).astype(float),axis=1) <= cap).all(), 'precommitted packet balls')
    return logits, np.tanh(logits)


def snapshot(body):
    return dict(state=source.vector(body._get_obs()).tolist(), block_velocity=list(map(float,body.block.velocity)),
        block_angular_velocity=float(body.block.angular_velocity), block_angle=float(body.block.angle))


class Rejected(ValueError):
    """A fixed candidate failed; retain denominator rather than replace it."""


class Store:
    def __init__(self, path):
        self.path = Path(path).absolute(); require(not self.path.exists() and shutil.disk_usage(self.path.parent).free >= FLOOR+CAP, 'NEW8MiB +200MiB floor')
        self.path.mkdir(); self.previous = None; self.index = 0
    def write(self, name, value, append=False, final=False):
        require(Path(name).name == name and name not in ('','.','..') and not (self.path/name).is_symlink(), 'flat nonsymlink artifact')
        used = sum(f.stat().st_size for f in self.path.iterdir() if f.is_file())
        require(used+len(value) <= CAP-(0 if final else FINAL) and shutil.disk_usage(self.path).free >= FLOOR+len(value), 'output cap /200MiB floor')
        with (self.path/name).open('ab' if append else 'xb') as f: f.write(value); f.flush(); os.fsync(f.fileno())
        fd = os.open(self.path,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
        return source.file_sha(self.path/name)
    def put(self,name,value,final=False): return self.write(name,raw(value),final=final)
    def emit(self,kind,**fields):
        e = dict(index=self.index, previous_sha256=self.previous, kind=kind, **deepcopy(fields))
        e['event_sha256'] = source.digest(e); self.write('journal.jsonl',raw(e),append=True)
        self.previous = e['event_sha256']; self.index += 1
    def arrays(self,name,**arrays):
        b = io.BytesIO(); np.savez_compressed(b,**arrays); return self.write(name,b.getvalue())
    def frame(self,name,value):
        from PIL import Image
        b = io.BytesIO(); Image.fromarray(value).save(b,format='PNG'); return self.write(name,b.getvalue())


class Ops:
    def __init__(self,p,store):
        self.p,self.store = p,store; self.started = time.monotonic(); self.counts = {k:dict(attempted=0,completed=0) for k in p['caps']}
    def call(self,kind,fn,**fields):
        require(time.monotonic()-self.started < self.p['wall_seconds'], 'wall cap before dispatch')
        c = self.counts[kind]; require(c['attempted'] < self.p['caps'][kind], 'predispatch '+kind+' cap')
        c['attempted'] += 1; self.store.emit('operation_attempted',resource=kind,**fields)
        v = fn(); c['completed'] += 1; self.store.emit('operation_completed',resource=kind,**fields); return v


def guard(rgb,state):
    s = source.vector(state); count = int(np.count_nonzero(np.all(rgb == [65,105,225],axis=-1)))
    if count == 0 or not ((s[:4] >= 0)&(s[:4] <= 512)).all(): raise Rejected('fresh RGB/canvas guard')
    return count


def attach_goal(body, goal, rgb):
    before = snapshot(body); body._set_goal_state(goal.copy()); body._goal = rgb.copy()
    require(snapshot(body) == before and np.array_equal(body.goal_state,goal) and np.array_equal(body._goal,rgb), 'goal attachment changed dynamics')


def episode(env,p,slot,phase,ops,store,reference=None):
    body = env.unwrapped
    _,info = ops.call('reset',lambda:env.reset(seed=slot['reset_seed']),slot=slot['index'],phase=phase,kind_of_reset='seed')
    original_goal = source.vector(info['goal_state'])
    requested = source.sample_state(original_goal,list(body.block.shapes),dict(root_seed=slot['root_seed']),0,0,p)
    store.emit('placement_recipe',slot=slot,phase=phase,original_goal=original_goal.tolist(),requested=requested.tolist())
    obs,info = ops.call('reset',lambda:env.reset(seed=0,options=dict(state=requested.copy(),goal_state=original_goal.copy())),slot=slot['index'],phase=phase,kind_of_reset='placement')
    history=[]; states=[]; dynamics=[]; future=[]; executed=[]; terminal=False; goal=None
    def capture(step):
        before = snapshot(body); rgb = image(env.render()); require(snapshot(body) == before and np.array_equal(obs['state'],body._get_obs()), 'render/observation changed dynamics')
        pixels = guard(rgb,obs['state']); name=f"s{slot['index']}_{phase}_{step:02}.png"
        hashed=store.frame(name,rgb)
        store.emit('fresh_observation',slot=slot['index'],phase=phase,step=step,state=source.vector(obs['state']).tolist(),snapshot=before,
            png=name,file_sha256=hashed,pixel_sha256=image_sha(rgb),canonical_agent_pixels=pixels)
        return rgb
    frame = capture(0)
    for step in range(1,11):
        if body.eval_state(original_goal,source.vector(obs['state']))[0]: raise Rejected('construction-recipe warmup already native terminal')
        if step == 1: history.append(frame); states.append(source.vector(obs['state'])); dynamics.append(snapshot(body))
        obs,reward,terminated,truncated,info = ops.call('public_step',lambda:env.step(np.zeros(2,np.float32)),phase=phase,step=step,raw_action=[0.,0.])
        frame=capture(step)
        if terminated or truncated: raise Rejected('warmup terminated/truncated; no continuation')
        if step in (5,10): history.append(frame); states.append(source.vector(obs['state'])); dynamics.append(snapshot(body))
    root = source.vector(obs['state']); logits,commands = witness(root,slot['sign'],p['packet_logit_cap'])
    # Durable command commitment BEFORE any witness action or replay dispatch.
    store.put(f"s{slot['index']}_{phase}_commit.json",dict(logits=logits.tolist(),commands=commands.tolist(),root_snapshot=snapshot(body)))
    if reference is not None:
        require(np.array_equal(reference['states'],states) and np.array_equal(reference['history'],history) and reference['dynamics'] == dynamics and np.array_equal(reference['commands'],commands), 'independent original recipe/history reconstruction')
        goal=reference['goal'].copy(); ops.call('goal_bind',lambda:attach_goal(body,goal,reference['goal_rgb']),phase=phase)
        if body.eval_state(goal,root)[0]: raise Rejected('derived goal already reached at root; retained no reach task')
    for i,command in enumerate(commands):
        obs,reward,terminated,truncated,info = ops.call('public_step',lambda c=command:env.step(c.copy()),phase=phase,step=11+i,raw_action=command.tolist())
        frame=capture(11+i); state=source.vector(obs['state']); future.append(state.copy()); executed.append(command.copy())
        if reference is not None:
            require(np.array_equal(state,reference['future'][i]) and np.array_equal(info['goal_state'],goal) and np.array_equal(info['goal'],reference['goal_rgb']), 'replay physics or goal changed')
            success,dist=body.eval_state(goal,state)
            require(bool(terminated) == bool(success) and float(reward) == -float(dist), 'native reward/termination unchanged')
        else: success=False
        store.emit('native_return',slot=slot['index'],phase=phase,step=i+1,terminated=bool(terminated),truncated=bool(truncated),reward=float(reward),state=state.tolist())
        if terminated or truncated:
            if reference is None or truncated or not success: raise Rejected('witness construction/verification terminated without native reach')
            terminal=True;break
    if reference is None:
        goal=source.vector(obs['state']); goal_rgb=frame.copy()
        certificate=f"s{slot['index']}_construction_certificate.npz"
        certificate_sha=store.arrays(certificate,goal=goal,goal_rgb=goal_rgb,witness_commands=commands,witness_logits=logits,
            future_states=np.asarray(future),observed_states=np.asarray(states),root_history=np.asarray(history),raw_action_history=np.zeros((10,2),np.float32))
        store.emit('construction_certificate_saved',file=certificate,sha256=certificate_sha)
    else:
        goal_rgb=reference['goal_rgb'].copy()
        require(terminal, 'fixed witness did not reach derived native goal within10steps')
        certificate=certificate_sha=None
    return dict(root=root,history=np.asarray(history),states=np.asarray(states),dynamics=dynamics,commands=commands,
        future=np.asarray(future),goal=goal,goal_rgb=goal_rgb,executed=np.asarray(executed),native_success=terminal,
        certificate=certificate,certificate_sha256=certificate_sha)


def validate_env(body):
    path=inspect.getsourcefile(type(body));require(path is not None and str(Path(path).resolve()) == str(ENV) and source.file_sha(path) == ENV_SHA, 'native class binding')
    require(body.relative is True and body.action_scale == 100 and body.dt == .01 and body.control_hz == 10 and body.with_target is False, 'new declared native rendering/action semantics')


def execute(p,store,factory,*,authorization):
    require(authorization == AUTH, 'explicit MAIN physics authorization'); verify(p)
    ops=Ops(p,store); rows=[dict(slot=deepcopy(s),status='not_started',reachable=False) for s in p['slots']]
    result=dict(schema=NAME+'.sourcepack',prepare_id=source.digest(p),rows=rows,errors=[],status='running',P1_admission=False,new_neural_calls=0,training=False,
        global_freshness=p['global_freshness'],direct_difficulty_required=False,hidden_state_scope='offline construction/evaluation only; no deployment selector')
    env=body=None;originals=[];space_hooks=[];in_reset=False
    def error(phase,exc): result['errors'].append(dict(phase=phase,type=type(exc).__name__,message=str(exc)))
    def install_space():
        space=body.space
        if any(x[0] is space for x in space_hooks):return
        owned='step' in space.__dict__; original=space.step
        space_hooks.append((space,owned,original)); space.step=lambda *a,**kw:ops.call('physics_step',lambda:original(*a,**kw))
    try:
        store.emit('run_started',prepare=p);env=factory();body=env.unwrapped;validate_env(body)
        old_set=body._set_state;old_render=body.render;old_reset=env.reset
        def setter(value):
            install_space();return ops.call('set_state',lambda:old_set(value))
        def render():return ops.call('goal_render' if in_reset else 'explicit_render',old_render)
        def reset(*a,**kw):
            nonlocal in_reset
            in_reset=True
            try:return old_reset(*a,**kw)
            finally:in_reset=False
        for obj,key,new in ((body,'_set_state',setter),(body,'render',render),(env,'reset',reset)):
            originals.append((obj,key,key in obj.__dict__,getattr(obj,key)));setattr(obj,key,new)
        for row,slot in zip(rows,p['slots'],strict=True):
            row['status']='running'
            try:
                construction=episode(env,p,slot,'construction',ops,store)
                replay0=episode(env,p,slot,'replay0',ops,store,construction)
                replay1=episode(env,p,slot,'replay1',ops,store,construction)
                root_hash=source.root_hash(replay0['root'],construction['goal'])
                require(root_hash not in p['exclusions']['root_hashes'] and root_hash not in [x.get('root_sha256') for x in rows], 'historical or within-batch duplicate root')
                payload=f"s{slot['index']}_blind_inputs.npz"
                hashed=store.arrays(payload,rgb_history=replay0['history'],raw_action_history=np.zeros((10,2),np.float32),goal_rgb=construction['goal_rgb'])
                row.update(status='reachable_native_replayed',reachable=True,root_sha256=root_hash,blind_payload=payload,blind_payload_sha256=hashed,
                    certificate=construction['certificate'],certificate_sha256=construction['certificate_sha256'],
                    replay_native_success=[replay0['native_success'],replay1['native_success']],executed_replay_steps=[len(replay0['executed']),len(replay1['executed'])],
                    object_displacement=float(np.linalg.norm(construction['goal'][2:4]-construction['root'][2:4])),
                    wrapped_object_rotation=float(abs(np.arctan2(np.sin(construction['goal'][4]-construction['root'][4]),np.cos(construction['goal'][4]-construction['root'][4])))))
            except Rejected as exc:row.update(status='rejected_fixed_candidate',reason=str(exc))
            store.emit('slot_finished',receipt=row)
        result['status']='fixed_slots_processed'
    except BaseException as exc:
        error('execution',exc);result['status']='engineering_failed'
        for row in rows:
            if row['status'] == 'running':row['status']='engineering_failed'
    finally:
        for obj,owned,original in reversed(space_hooks):
            try:
                if owned:obj.step=original
                else:delattr(obj,'step')
            except BaseException as exc:error('restore_space',exc)
        for obj,key,owned,original in reversed(originals):
            try:
                if owned:setattr(obj,key,original)
                else:delattr(obj,key)
            except BaseException as exc:error('restore_hook',exc)
        if env is not None:
            try:env.close()
            except BaseException as exc:error('close',exc)
        try:verify(p)
        except BaseException as exc:error('final_seal',exc)
        if not result['errors']:
            try:
                c=ops.counts
                require(all(v['attempted'] == v['completed'] for v in c.values()) and
                    c['physics_step']['completed'] == 2*c['reset']['completed']+10*c['public_step']['completed'] and
                    c['goal_render']['completed'] == c['reset']['completed'] and c['set_state']['completed'] == 2*c['reset']['completed'], 'actual completed internal physics/reset census')
            except BaseException as exc:error('final_census',exc)
        result.update(counts=ops.counts,first_error=deepcopy(result['errors'][0]) if result['errors'] else None,
            reachable_count=sum(x['reachable'] for x in rows),new_independent_confirmation=False,
            physics_note='actual space.step hooks include partial operations; no hidden state restore',wall_seconds=time.monotonic()-ops.started)
        if result['errors']:result['status']='engineering_failed'
        try:store.emit('run_finished',status=result['status'],counts=ops.counts,errors=result['errors'])
        except BaseException as exc:error('final_journal',exc);result['status']='engineering_failed'
        result['journal_chain_tail']=store.previous
        result['first_error']=deepcopy(result['errors'][0]) if result['errors'] else None
        result['sourcepack_complete']=result['status'] == 'fixed_slots_processed' and not result['errors']
        try:store.put('sourcepack.json',result,final=True)
        except BaseException as exc:
            error('final_report',exc);result['status']='engineering_failed';result['sourcepack_complete']=False;result['first_error']=deepcopy(result['errors'][0])
    return result


def load_blind(path,expected_sha):
    require(source.file_sha(path) == expected_sha, 'blind payload external SHA')
    with np.load(path,allow_pickle=False) as z:
        require(set(z.files) == {'rgb_history','raw_action_history','goal_rgb'}, 'no witness or privileged data in deployment input')
        values={k:z[k].copy() for k in z.files}
    require(values['rgb_history'].shape == (3,224,224,3) and values['rgb_history'].dtype == np.uint8, 'actual observed RGB3')
    image(values['goal_rgb']);a=values['raw_action_history']
    require(a.shape == (10,2) and a.dtype == np.float32 and not a.any(), 'actual zero warmup history')
    return values


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run'))
    parser.add_argument('--prepared',type=Path);parser.add_argument('--prepare-sha');parser.add_argument('--output',type=Path)
    parser.add_argument('--root-seeds',type=int,nargs='+');parser.add_argument('--reset-seeds',type=int,nargs='+');parser.add_argument('--wall-seconds',type=int)
    parser.add_argument('--authorize-physics',action='store_true');a=parser.parse_args()
    if a.command == 'prepare':value=prepare(a.output,root_seeds=a.root_seeds,reset_seeds=a.reset_seeds,wall_seconds=a.wall_seconds)
    else:
        require(source.file_sha(a.prepared) == a.prepare_sha, 'external prepare SHA');p=read_prepared(a.prepared.read_bytes())
        if a.command == 'verify':value=p['schema']
        else:
            require(a.authorize_physics, 'MAIN authorization required')
            def factory():
                import gymnasium as gym
                import stable_worldmodel  # MAIN-only registration, no model.
                return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False)
            store=Store(a.output);store.put('execution_binding.json',dict(prepared=str(a.prepared.resolve()),prepare_file_sha256=a.prepare_sha))
            value=execute(p,store,factory,authorization=AUTH)
            print(source.canonical(dict(status=value['status'],reachable_count=value['reachable_count'],counts=value['counts'])))
            if value['status'] == 'engineering_failed':raise SystemExit(1)
            return
    print(source.canonical(value))


if __name__ == '__main__':main()
