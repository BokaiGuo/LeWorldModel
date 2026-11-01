"""Four preregistered native source slots; MAIN-only physics, never neural.

Reuses frozen moving.episode, NOT its two-slot build/execute or globals.
calibration_only is explicitly nonfresh; fresh_candidate needs approved history.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from types import ModuleType
import numpy as np
from experiments.spectral_reachability import p1_constructive_moving_history_source_20261027_v1 as moving
from experiments.spectral_reachability import p1_moving_live_feedback_physical_20261027_v1 as live
from experiments.spectral_reachability import moving_root_exclusion_manifest_20261027_v1 as exclusion

NAME = 'agent_p4_fresh_source_20261028_v1'
ROOT = moving.ROOT
AUTH = 'MAIN_FOUR_SLOT_NATIVE_SOURCE_PHYSICS_ONLY'
REG = ROOT.parent/'docs/chat_freshroots_20261027_preregistration.json'
REG_SHA = '79e1f45e79d692cae72376c91a4830c94b6ac1f53bb346eeaa085715e4197d7e'
PROTOCOL = ROOT.parent/'docs/D2P_MOVING_FRESH_ROOT_PROTOCOL_20261027.md'
PROTOCOL_SHA = '42f1bcbf517dfd46b8ba05525663d82efac32dbb5fb24f87a8ca572bef507a2f'
MOVING_SHA = '9fb76ecfa054f878ab9778fc29c59bc708728b240dd2cbc3060b166e52622f55'
LIVE_SHA = '6742a1b35102e433d7772e4a208b4f97b37b76f815b991cfc0e472763754ace9'
CAP, FLOOR, FINAL = 16*1024**2, 200*1024**2, 512*1024
require, sha, raw = moving.require, moving.source.file_sha, moving.raw


def slots():
    return [dict(index=i,root_seed=203027101+i,reset_seed=203027201+i,sign=1 if i%2 == 0 else -1)
            for i in range(4)]


def read(path, expected=None, cap=4*1024**2):
    p = Path(path).absolute()
    require(p.resolve() == p and not p.is_symlink() and p.is_file() and p.stat().st_size <= cap,
            'bounded canonical regular metadata')
    payload = p.read_bytes()
    h=hashlib.sha256(payload).hexdigest()
    require(len(payload) <= cap and sha(p) == h and (expected is None or h == expected), 'external metadata byte SHA')
    def pairs(items):
        result = {}
        for k,v in items:
            require(k not in result, 'duplicate JSON key'); result[k] = v
        return result
    out = json.loads(payload,object_pairs_hook=pairs)
    moving.source.canonical(out)  # Reject NaN/Infinity recursively.
    return out


def history_certificate(path, expected):
    """Existing exclusion schema + its external MAIN content approval.

    SHA-bound approval is a MAIN-supplied authority input, not a signature or
    a proof that this adapter can recover unsaved historical executions.
    """
    require(path is not None and expected is not None, 'fresh_candidate requires external historical certificate')
    c = read(path,expected)
    require(c['schema'] == 'moving_fresh_source_exclusions.v1' and c['protocol']['sha256'] == PROTOCOL_SHA,
            'existing history certificate/protocol')
    require(c['source_run_preconditions_met'] is True and c['unresolved_sources'] == [] and
            c['approval_errors'] == [] and c['approval']['approved'] is True,
            'complete original reset/history coverage required; unknown is not fresh')
    extra = {'coverage_content_sha256','approval','approval_errors','source_run_preconditions_met','status',
             'request_sha256','projected_file_bytes','expanded_metadata_bytes'}
    content = {k:v for k,v in c.items() if k not in extra}
    require(exclusion.digest(content) == c['coverage_content_sha256'], 'recomputed exclusion content SHA')
    approval = c['approval']['external_file']
    a = read(approval['path'],approval['sha256'])
    require(a['schema'] == exclusion.NAME+'.MAIN_coverage_approval' and a['authority'] == 'MAIN' and
            a['approved'] is True and a['manifest_content_sha256'] == c['coverage_content_sha256'] and
            a['cutoff_utc'] == c['cutoff_utc'] and
            set(a['coverage_scope']) >= set(exclusion.REQUIRED_FAMILIES) and
            set(c['approval']['coverage_scope']) == set(a['coverage_scope']), 'external MAIN exact history approval')
    require(c['project_root'] == str(ROOT) and c['observed_roots'], 'nonempty complete physical history')
    require(set(e['family'] for e in c['evidence_files']) >= set(exclusion.REQUIRED_FAMILIES), 'all historical source families')
    for path,h in c['source_sha256'].items():require(sha(path) == h, 'history projector source closure')
    require(c['source_sha256'].get(str(Path(exclusion.__file__).resolve())) == sha(exclusion.__file__), 'original history projector binding')
    for r in c['observed_roots']:
        identity=exclusion.state_identity(r['root_state'])
        require(all(r.get(k) is not None for k in ('root_state_sha256','physical_pose_key','root_snapshot_sha256',
                    'history_pixels_sha256','raw_history_sha256')), 'complete historical actual root identity')
        require(identity['root_state_sha256'] == r['root_state_sha256'] and identity['physical_pose_key'] == r['physical_pose_key'],
                'historical state/pose hashes agree with actual state')
        require(len(r['history_pixels_sha256']) == 3, 'three actual historical RGB identities')
        for h in [r['root_snapshot_sha256'],r['raw_history_sha256'],*r['history_pixels_sha256']]:exclusion.hash_value(h)
    return c


def module_closure():
    """Bind already imported experiment modules, never import/run new assets."""
    seen={};todo=[moving,live,exclusion]
    while todo:
        module=todo.pop()
        if module.__name__ in seen:continue
        require(len(seen) < 128, 'bounded imported source closure')
        seen[module.__name__]=str(Path(module.__file__).resolve())
        todo.extend(v for v in vars(module).values() if isinstance(v,ModuleType) and
                    v.__name__.startswith('experiments.') and getattr(v,'__file__',None))
    return sorted(set(seen.values()))


def seed_check(plan_slots, seeds, spans):
    for s in plan_slots:
        for key in ('root_seed','reset_seed'):
            v=s[key]
            require(v not in seeds and all(not lo <= v < hi for lo,hi in spans),
                    'fixed seed collision; never replace or retry')


def build(mode, wall_seconds=1200, history_cert=None, history_sha=None):
    require(mode in ('calibration_only','fresh_candidate'), 'explicit collection mode')
    require(type(wall_seconds) is int and 0 < wall_seconds <= 1200, 'whole job wall1..1200')
    require(sha(REG) == REG_SHA and sha(PROTOCOL) == PROTOCOL_SHA and
            sha(moving.__file__) == MOVING_SHA and sha(live.__file__) == LIVE_SHA, 'frozen protocol and native machinery')
    registration = read(REG,REG_SHA)
    expected = [dict(slot=s['index'],root_seed=s['root_seed'],reset_seed=s['reset_seed'],witness_sign=s['sign']) for s in slots()]
    require(registration['candidate_order'] == expected and registration['sampling_denominator'] == 4 and
            registration['replacement_or_retry'] is False, 'exact four preregistered slots')
    require(sha(moving.RECIPE_PREPARED) == moving.RECIPE_SHA, 'original recipe prepare external SHA')
    original = moving.old.read_prepared(Path(moving.RECIPE_PREPARED).read_bytes())
    cert = history_certificate(history_cert,history_sha) if history_cert is not None or mode == 'fresh_candidate' else None
    require(history_cert is not None or history_sha is None, 'no orphan history SHA')
    seeds = [x['seed'] for x in original['exclusions']['seed_families']]
    spans = [x['span'] for x in original['exclusions']['reset_spans']]
    if cert is not None:
        seeds += cert['used_or_reserved_root_seeds']+cert['used_or_reserved_reset_seeds']
        spans += cert['reserved_reset_spans']
    seed_check(slots(),seeds,spans)
    bindings = deepcopy(original['bindings'])
    files = [Path(__file__).resolve(),ROOT/'tests'/(NAME+'_tests.py'),ROOT/'innerdocs'/(NAME+'.md'),
             Path(moving.__file__).resolve(),Path(live.__file__).resolve(),Path(exclusion.__file__).resolve(),
             Path(moving.RECIPE_PREPARED),REG,PROTOCOL]
    for path in files: bindings[str(path)] = sha(path)
    for path in module_closure():bindings[path]=sha(path)
    if cert is not None:
        bindings[str(Path(history_cert).absolute())] = history_sha
        a = cert['approval']['external_file'];bindings[a['path']] = a['sha256']
        for e in cert['evidence_files']:bindings[e['path']] = e['sha256']
        bindings.update(cert['source_sha256'])
    for path,h in bindings.items():require(sha(path) == h, 'full original recipe/source/native/input closure: '+path)
    return dict(schema=NAME+'.prepare',mode=mode,slots=slots(),fixed_denominator=4,recipe=deepcopy(original['recipe']),
                witness=original['witness'],packet_logit_cap=original['packet_logit_cap'],environment=original['environment'],
                with_target=False,raw_warmup=moving.warmup_commands().tolist(),warmup=10,horizon=10,replay_episodes=2,
                wall_seconds=wall_seconds,caps=dict(constructor=1,reset=24,set_state=48,goal_render=24,
                    goal_setter=32,goal_bind=8,public_step=240,physics_step=2448,explicit_render=252),
                output_cap_bytes=CAP,disk_floor_bytes=FLOOR,final_reserve_bytes=FINAL,
                original_recipe=dict(path=str(moving.RECIPE_PREPARED),sha256=moving.RECIPE_SHA),
                history_certificate=None if cert is None else dict(path=str(Path(history_cert).absolute()),sha256=history_sha),
                historical_roots=[] if cert is None else deepcopy(cert['observed_roots']),
                root_goal_exclusions=sorted(set(original['exclusions']['root_hashes']+
                    ([] if cert is None else cert['root_goal_sha256s']))),bindings=bindings,
                freshness=False,confirmed=False,P1_admission=False,P4_admission=False,neural_calls=0,training=False,
                historical_coverage='unknown' if cert is None else 'external_MAIN_approved',
                source_only=True,solver_cache_identity_verified=False,global_access_log_required=False)


def verify(p):
    c = p['history_certificate']
    require(p == build(p['mode'],p['wall_seconds'],None if c is None else c['path'],None if c is None else c['sha256']),
            'complete four-slot adapter reconstruction')
    return p


class Store(moving.Store):
    def __init__(self,path):
        self.path=Path(path).absolute()
        require(self.path.parent == ROOT/'results' and self.path.name.startswith(NAME) and
                not self.path.exists() and self.path.resolve() == self.path, 'new own source output')
        require(shutil.disk_usage(self.path.parent).free >= FLOOR+CAP, '16MiB output +200MiB floor')
        self.path.mkdir();self.previous=None;self.index=0

    def check(self):
        require(sum(p.stat().st_size for p in self.path.iterdir()) <= CAP-FINAL and
                shutil.disk_usage(self.path).free >= FLOOR+FINAL, 'output/final/disk reserve before operation')

    def file_sha(self,name):return sha(self.path/name)

    def write(self,name,value,append=False,final=False):
        require(Path(name).name == name and name not in ('','.','..') and not (self.path/name).is_symlink(), 'flat own artifact')
        used=sum(p.stat().st_size for p in self.path.iterdir())
        require(used+len(value) <= CAP-(0 if final else FINAL) and
                shutil.disk_usage(self.path).free >= FLOOR+len(value), 'bounded durable output')
        with (self.path/name).open('ab' if append else 'xb') as f:
            require(f.write(value) == len(value), 'short write');f.flush();os.fsync(f.fileno())
        fd=os.open(self.path,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        return sha(self.path/name)


class Meter(live.Meter):
    def __init__(self,p,store,started):
        super().__init__(store);self.p=p;self.started=started;self.caps=deepcopy(p['caps'])
        self.counts={k:dict(attempted=0,completed=0) for k in self.caps}

    def check(self):
        require(time.monotonic()-self.started < self.p['wall_seconds'], 'whole job source wall cap')
        self.store.check()


class EpisodeOps:
    """Old episode outer API -> actually metered native virtual methods.

    reset/public_step are charged only by frozen native subclass, not twice.
    """
    def __init__(self,meter,body):self.meter=meter;self.body=body
    def call(self,kind,fn,**fields):
        meter=self.meter
        if kind in ('reset','public_step'):
            meter.region='setup' if kind == 'reset' else ('warmup' if fields['step'] <= 10 else 'suffix')
            before=meter.counts[kind]['completed'];v=fn()
            require(meter.counts[kind]['completed'] == before+1, 'one actual native return per outer source operation')
            return v
        require(kind == 'goal_bind', 'only native source operations')
        meter.region='goal_bind';before=moving.snapshot(self.body);pose=np.asarray(self.body.goal_pose).copy()
        v=meter.call('goal_bind',fn)
        require(moving.snapshot(self.body) == before and np.array_equal(pose,self.body.goal_pose), 'bind never changes state/goal_pose')
        return v


def root_identity(observation,history):
    state=moving.source.vector(observation['state'])
    pose=np.concatenate((state[:4],[np.remainder(state[4],2*np.pi)]));pose[pose == 0]=0.
    return dict(root_state=state.tolist(),physical_pose=pose.tolist(),
                physical_pose_key=moving.image_sha(np.asarray(pose,dtype='<f8')),
                root_state_sha256=moving.image_sha(np.asarray(state,dtype='<f8')),
                root_snapshot_sha256=moving.source.digest(observation['snapshot']),
                history_pixels_sha256=[x['pixel_sha256'] for x in history],
                raw_history_sha256=moving.image_sha(moving.warmup_commands()))


def duplicate(a,b):
    state=moving.source.vector(b['root_state'])
    pose=np.concatenate((state[:4],[np.remainder(state[4],2*np.pi)]))
    return (np.all(np.abs(np.asarray(a['physical_pose'])[:4]-pose[:4]) <= 1e-9) and
            abs(np.arctan2(np.sin(a['physical_pose'][4]-pose[4]),np.cos(a['physical_pose'][4]-pose[4]))) <= 1e-12) or any(
                a[k] == b.get(k) for k in ('root_state_sha256','root_snapshot_sha256','history_pixels_sha256'))


class GateStore:
    """Concrete pre-witness gate inserted at saved actual warmup10 guard return."""
    def __init__(self,store,p,row,batch):
        self.store=store;self.p=p;self.row=row;self.batch=batch;self.frames={};self.roots={}
    def __getattr__(self,key):return getattr(self.store,key)
    def emit(self,kind,**fields):
        self.store.emit(kind,**fields)
        if kind == 'fresh_observation' and fields['step'] in (0,5,10):
            self.frames[(fields['phase'],fields['step'])]=deepcopy(fields)
        if kind != 'fresh_guard_passed' or fields['step'] != 10:return
        phase=fields['phase'];history=[self.frames[(phase,b)] for b in (0,5,10)]
        identity=root_identity(history[-1],history)
        if phase == 'construction':
            self.row['actual_root_identity']=identity
            collision=any(duplicate(identity,h) for h in self.p['historical_roots']+self.batch)
            # Include observed roots of unsuccessful slots in batch exclusions.
            self.batch.append(identity)
            self.store.emit('actual_root_pre_witness_check',slot=fields['slot'],phase=phase,identity=identity,
                            duplicate=collision,historical_coverage=self.p['historical_coverage'])
            if collision:raise moving.Rejected('historical/within-batch physical root duplicate before witness')
            self.roots[phase]=identity
        else:
            require(identity == self.roots['construction'], 'same actual native root/history before replay witness')
            self.store.emit('replay_root_pre_witness_check',slot=fields['slot'],phase=phase,identity=identity)


def execute(p,store,factory,*,authorization,started=None):
    require(authorization == AUTH, 'explicit MAIN native physics authorization');verify(p)
    meter=Meter(p,store,time.monotonic() if started is None else started)
    rows=[dict(slot=deepcopy(s),status='not_started',reachable=False,blind_payload=None,
               freshness=False,confirmed=False,actual_root_identity=None,
               phases=[dict(phase=phase,status='not_started') for phase in ('construction','replay0','replay1')]) for s in slots()]
    result=dict(schema=NAME+'.sourcepack',prepare_id=moving.source.digest(p),mode=p['mode'],rows=rows,
                fixed_denominator=4,errors=[],freshness=False,confirmed=False,P1_admission=False,P4_admission=False,
                neural_calls=0,training=False,status='running',sourcepack_complete=False)
    env=None;batch=[]
    def error(stage,exc):result['errors'].append(dict(stage=stage,type=type(exc).__name__,message=str(exc)))
    try:
        store.put('roster.json',dict(rows=rows,fixed_denominator=4));store.emit('run_started',prepare_id=result['prepare_id'],mode=p['mode'])
        meter.episode='source_constructor';env=meter.call('constructor',lambda:factory(meter));body=env.unwrapped
        require(body._live_meter is meter and body._live_constructor_complete is True and body.relative is True and
                body.with_target is False and body.render_action is False and body.render_size == 224 and
                body.action_scale == 100 and body.control_hz == 10 and body.dt == .01, 'native constructor/invariants')
        ops=EpisodeOps(meter,body)
        for row,slot in zip(rows,p['slots'],strict=True):
            row['status']='running';gated=GateStore(store,p,row,batch);values=[]
            try:
                for phase in ('construction','replay0','replay1'):
                    meter.episode=f"s{slot['index']}_{phase}";meter.region='setup'
                    ph=next(ph for ph in row['phases'] if ph['phase'] == phase);ph['status']='attempted'
                    store.emit('phase_attempt',slot=slot['index'],phase=phase)
                    before=deepcopy(meter.counts);pose=np.asarray(body.goal_pose).copy()
                    value=moving.episode(env,p,slot,phase,ops,gated,None if not values else values[0])
                    require(np.array_equal(pose,body.goal_pose), 'renderer outline goal_pose never modified')
                    if phase == 'construction':
                        h=moving.source.root_hash(value['root'],value['goal'])
                        if h in p['root_goal_exclusions']:raise moving.Rejected('historical root-goal duplicate')
                        row['root_sha256']=h
                    ph.update(status='returned',public_steps=10+len(value['executed']),native_success=value['native_success'],
                              counts_delta={k:{c:meter.counts[k][c]-before[k][c] for c in ('attempted','completed')} for k in meter.counts})
                    store.emit('phase_return',slot=slot['index'],receipt=ph);values.append(value)
                construction,replay0,replay1=values
                payload=f"s{slot['index']}_blind_inputs.npz"
                h=store.arrays(payload,rgb_history=replay0['history'],raw_action_history=replay0['raw_history'],goal_rgb=construction['goal_rgb'])
                row.update(status='reachable_native_replayed',reachable=True,blind_payload=payload,blind_payload_sha256=h,
                           certificate=construction['certificate'],certificate_sha256=construction['certificate_sha256'],
                           replay_native_success=[replay0['native_success'],replay1['native_success']],
                           executed_replay_steps=[len(replay0['executed']),len(replay1['executed'])],
                           freshness=p['mode'] == 'fresh_candidate',confirmed=False)
            except moving.Rejected as exc:
                row.update(status='rejected_fixed_candidate',reason=str(exc))
                for ph in row['phases']:
                    if ph['status'] == 'attempted':ph.update(status='rejected',reason=str(exc))
            except BaseException as exc:
                for ph in row['phases']:
                    if ph['status'] == 'attempted':ph.update(status='failed_pending',reason=str(exc))
                raise
            store.emit('slot_finished',receipt=row)
        result['status']='fixed_slots_processed'
    except BaseException as exc:
        error('execution',exc);result['status']='engineering_failed'
        for row in rows:
            if row['status'] == 'running':row.update(status='engineering_failed',reason=str(exc))
    finally:
        meter.region='closing'
        if env is not None:
            try:env.close()
            except BaseException as exc:error('close_or_restore',exc)
        result['errors'].extend(meter.secondary)
        try:verify(p)
        except BaseException as exc:error('final_closure',exc)
        try:
            c=meter.counts
            require(not meter.pending and all(x['attempted'] == x['completed'] for x in c.values()), 'pending actual operation retained')
            require(c['physics_step']['completed'] == 2*c['reset']['completed']+10*c['public_step']['completed'] and
                    c['set_state']['completed'] == 2*c['reset']['completed'] and c['goal_render']['completed'] == c['reset']['completed'] and
                    c['goal_setter']['completed'] == c['reset']['completed']+c['goal_bind']['completed'], 'full native source census')
            meter.check()
        except BaseException as exc:error('final_census',exc)
        if result['errors']:result['status']='engineering_failed'
        result.update(counts=meter.counts,pending=deepcopy(meter.pending),reachable_count=sum(r['reachable'] for r in rows),
                      freshness=p['mode'] == 'fresh_candidate' and all(r['reachable'] for r in rows) and not result['errors'],
                      wall_seconds=time.monotonic()-meter.started,first_error=result['errors'][0] if result['errors'] else None)
        try:
            store.emit('run_finished',status=result['status'],counts=meter.counts,pending=meter.pending,rows=rows)
            result['journal_chain_tail']=store.previous
            result['journal_sha256']=store.file_sha('journal.jsonl')
            result['sourcepack_complete']=result['status'] == 'fixed_slots_processed' and not result['errors']
            store.put('sourcepack.json',result,final=True)
        except BaseException as exc:
            error('final_persistence',exc);result.update(status='engineering_failed',sourcepack_complete=False,freshness=False)
            result['first_error']=result['errors'][0]
    return result


def prepare(output,**kwargs):
    p=build(**kwargs);path=Path(output).absolute();data=raw(p)
    require(path.parent == ROOT/'results' and path.name.startswith(NAME) and not path.exists() and len(data) <= 256*1024,
            'new own bounded256KiB prepare')
    require(shutil.disk_usage(path.parent).free >= FLOOR+len(data), 'prepare disk floor')
    with path.open('xb') as f:require(f.write(data) == len(data), 'short prepare');f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(prepared=str(path),prepare_sha256=sha(path),mode=p['mode'],freshness=False,confirmed=False)


def load_prepared(path,expected):return verify(read(path,expected,256*1024))


def run(prepared,expected,output,*,authorization):
    started=time.monotonic();require(authorization == AUTH, 'MAIN-only source physics')
    p=load_prepared(prepared,expected);store=Store(output)
    store.put('execution_binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
    return execute(p,store,live.native_factory,authorization=authorization,started=started)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run'))
    parser.add_argument('--mode',choices=('fresh_candidate','calibration_only'))
    for key in ('output','prepared','prepare-sha','history-cert','history-sha'):parser.add_argument('--'+key)
    parser.add_argument('--wall-seconds',type=int,default=1200);parser.add_argument('--authorize-main-physics',action='store_true')
    a=parser.parse_args()
    if a.command == 'prepare':out=prepare(a.output,mode=a.mode,wall_seconds=a.wall_seconds,history_cert=a.history_cert,history_sha=a.history_sha)
    elif a.command == 'verify':out=dict(schema=load_prepared(a.prepared,a.prepare_sha)['schema'],verified=True,physical_release=False)
    else:
        require(a.authorize_main_physics, 'explicit MAIN physics flag')
        out=run(a.prepared,a.prepare_sha,a.output,authorization=AUTH)
    print(moving.source.canonical(out))
    if a.command == 'run' and not out['sourcepack_complete']:raise SystemExit(2)


if __name__ == '__main__':main()
