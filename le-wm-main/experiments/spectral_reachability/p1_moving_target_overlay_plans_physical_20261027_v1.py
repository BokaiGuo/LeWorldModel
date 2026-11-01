"""MAIN-only fixed moving-plan replay; native off/on/off is display-only.

No model, controller selection, witness execution, goal_pose edit, or old execute.
The native default outline is NOT the constructed task goal. Goal-on production
certification belongs to a separately SHA-bound source/consumer bridge.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import io
import os
from pathlib import Path
import shutil
import time
import numpy as np

from experiments.spectral_reachability import p1_constructive_moving_history_physical_20261027_v1 as prior

NAME = 'p1_moving_target_overlay_plans_physical_20261027_v1'
AUTH = 'MAIN_MOVING_TARGET_OVERLAY_PLANS_PHYSICS_ONLY'
ROOT, low, moving, source = prior.ROOT, prior.low, prior.moving, prior.source
require, sha, raw, read = prior.require, prior.sha, prior.raw, prior.read
REPORT = ROOT/'results/p1_constructive_moving_history_physical_20261027_v1_main01/report.json'
REPORT_SHA = 'a5322dc4f483e82b5ed9d7b3996cf8a63845261d65774f90a4ae341773016aaf'
PREPARE_SHA = '226749e2bf8acdcd7a4a4a08be7336445daad0b6ad12185b24cbc0fb4560b1ff'
PRIOR_SHA = '44d90599eab881c773d5efa09177dfe6b395fa3f4eef41070340681fea5ec96e'
SOURCEPACK_SHA = 'c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6'
ARMS = ('line_search_GN', 'full10_conditional_D2P')
CAP, FLOOR, FINAL = 8*1024**2, 200*1024**2, 256*1024


def contract():
    return dict(fixed_denominator=4, root_slots=2, arms=list(ARMS), warmup=10, horizon=10,
        checkpoints=[0, 5, 10], flags=[False, True, False], with_target=False,
        wall_seconds=300, caps=dict(reset=8, public_step=80, physics_step=816,
            explicit_render=112, goal_render=8, set_state=16, goal_bind=4),
        output_cap_bytes=CAP, disk_floor_bytes=FLOOR, final_reserve_bytes=FINAL,
        PNG='88 original off + 12*(on,off); triplet first receipt is original off, never an extra render',
        intervention='with_target only; unchanged constructed goal_state/off goalRGB and native goal_pose',
        on_semantics='native default outline; NOT constructed task goal; no source goal-on certification here',
        stop='native terminal/truncation honored; missing checkpoints retained; scientific denominator always4',
        scope='known moving roots/calibration only; no new selection, independent samples, model, fitting, training or P1 admission')


def png(path, expected):
    from PIL import Image
    require(sha(path) == expected and Path(path).stat().st_size <= 1024**2, 'sealed bounded PNG')
    value = Path(path).read_bytes()
    with Image.open(io.BytesIO(value)) as im:
        require(im.format == 'PNG' and im.mode == 'RGB' and im.size == (224,224), 'literal RGB224 PNG')
        image = np.asarray(im).copy()
    require(hashlib.sha256(value).hexdigest() == expected and sha(path) == expected, 'PNG read seal')
    return image


def collect_saved(p, report, events, directory):
    """Saved old off references only; control commands come solely from prepare."""
    keys = [(i, arm) for i in range(2) for arm in ARMS]
    require([(x['root_slot'],x['arm']) for x in p['plans']] == keys and
        [(x['root_slot'],x['arm']) for x in report['rows']] == keys, 'fixed moving four-arm identities')
    require(report['schema'] == prior.NAME+'.report' and report['status'] == 'moving_history_physical_recorded'
        and report['errors'] == [] and report['sourcepack_sha256'] == p['sourcepack_sha256'] == SOURCEPACK_SHA,
        'complete actual moving source, never zero-history report')
    require(events[0]['kind'] == 'run_started' and events[0]['prepare'] == p and
        events[-1]['counts'] == report['counts'] and events[-1]['errors'] == [], 'old start/finish binding')
    saved, bindings = [], {}
    for i, (plan, row) in enumerate(zip(p['plans'], report['rows'], strict=True)):
        require(row['selection_digest'] == plan['selection_digest'] and row['status'] == 'horizon_completed' and
            row['full_horizon'] is True and row['initial_exact'] is True and row['first_error'] is None and
            row['secondary_errors'] == [] and row['own_returned'] == row['own_attempted'] == row['warmup_returned'] == 10,
            'old fixed full10 row and selected identity')
        branch_events = [e for e in events if e.get('branch') == i]
        starts = [e for e in branch_events if e['kind'] == 'branch_started']
        require(len(starts) == 1 and starts[0]['plan'] == plan, 'old selected own commands commitment')
        observations = [e['receipt'] for e in branch_events if e['kind'] == 'fresh_observation']
        expected = [(phase, step) for phase in ('warmup','plan') for step in range(11)]
        require([(r['phase'],r['step']) for r in observations] == expected and row['PNG'] == 22,
            'all22 original off observations, not only checkpoints')
        off = {}
        for r in observations:
            name = r['png']; require(type(name) is str and Path(name).name == name, 'flat old PNG')
            path = Path(directory)/name; a = png(path, r['file_sha256'])
            require(moving.image_sha(a) == r['pixel_sha256'] and r['snapshot']['state'] == r['state'] == r['observation_state'],
                'old off PNG/state/snapshot receipts')
            bindings[str(path.resolve())] = r['file_sha256']; off[(r['phase'],r['step'])] = deepcopy(r)
        returns = [e for e in branch_events if e['kind'] == 'returned_transition']
        require([(e['phase'],e['step']) for e in returns] == [(ph,s) for ph in ('warmup','plan') for s in range(1,11)],
            'old full transition chronology')
        actions = low.commands(plan['commands'])
        for e, a in zip([e for e in returns if e['phase'] == 'plan'], actions, strict=True):
            require(moving.exact(np.asarray(e['command'],np.float32),a), 'old returned own raw command')
        warms = [e for e in branch_events if e['kind'] == 'warmup_return']
        require([e['step'] for e in warms] == list(range(1,11)) and
            [e['raw_float32_bits'] for e in warms] == [a.tobytes().hex() for a in moving.warmup_commands()],
            'old moving nonzero raw10 history')
        require(row['checkpoints'] == [off[('plan',s)] for s in (0,5,10)], 'old checkpoint pointers')
        saved.append(dict(off=off, transitions={(e['phase'],e['step']):deepcopy(e) for e in returns}))
    return saved, bindings


def inputs():
    require(sha(prior.__file__) == PRIOR_SHA and sha(REPORT) == REPORT_SHA, 'fixed original moving producer/report')
    report = read(REPORT); bp = REPORT.parent/'execution_binding.json'; binding = read(bp)
    require(binding['prepare_sha256'] == PREPARE_SHA, 'external original moving prepare SHA')
    pp = Path(binding['prepared']).resolve(); p = prior.verify(pp, PREPARE_SHA)  # saved-only reconstruction
    m = read(p['manifest'])
    data = prior.load_source(m['sourcepack'],m['sourcepack_sha256'],m['source_prepared'],m['source_prepare_sha256'])
    jp = REPORT.parent/'journal.jsonl'; require(sha(jp) == report['journal_sha256'], 'old journal file SHA')
    events = low.journal(jp, report['journal_chain_tail'])
    saved, frames = collect_saved(p, report, events, REPORT.parent)
    bindings = {**p['bindings'], **data['bindings'], **frames, str(REPORT):REPORT_SHA,
        str(bp):sha(bp), str(pp):PREPARE_SHA, str(jp):report['journal_sha256']}
    for path,h in bindings.items(): require(sha(path) == h, 'old moving closure '+path)
    return dict(plans=deepcopy(p['plans']), source_slots=deepcopy(p['source_slots']),
        original_report_sha256=REPORT_SHA, original_prepare_sha256=PREPARE_SHA,
        sourcepack_sha256=SOURCEPACK_SHA, bindings=bindings), data, saved


def build():
    value, _, _ = inputs(); bindings = deepcopy(value['bindings'])
    for path in (Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')):
        bindings[str(path)] = sha(path)
    return dict(schema=NAME+'.prepare', contract=contract(), source=value, bindings=bindings,
        fixed_denominator=4, P1_admission=False, source_goal_on_authority='external SHA bridge required; not produced or read here')


def prepare(output):
    p = build(); value = raw(p); path = Path(output).absolute()
    require(len(value) < 256*1024 and not path.exists() and shutil.disk_usage(path.parent).free >= FLOOR+len(value)+4096,
        'exclusive small metadata prepare /200MiB floor')
    with path.open('xb') as f: require(f.write(value) == len(value), 'short prepare'); f.flush(); os.fsync(f.fileno())
    fd = os.open(path.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(value))


def verify(path, expected):
    require(sha(path) == expected, 'external overlay prepare SHA'); p = read(path)
    require(p == build(), 'full own fixed contract and closure reconstruction')
    return p


class Store(low.Store):
    """Inherited flat exclusive 8MiB/200MiB durable writer; no globals patched."""


class Ops(low.Ops):
    def __init__(self,p,store):
        super().__init__(p,store)
        self.active={k:0 for k in self.counts}

    def call(self, kind, fn, **fields):
        def dispatch():
            self.active[kind]+=1
            try: return fn()
            finally: self.active[kind]-=1
        try: return super().call(kind,dispatch,**fields)
        except BaseException as exc:
            try: self.store.emit('operation_failed',resource=kind,**self.where,**fields,
                exception=dict(type=type(exc).__name__,message=str(exc)))
            except BaseException:  # Never replace the original dispatch/journal error.
                raise exc
            raise


def literal(v):
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,np.generic): return v.item()
    if isinstance(v,dict): return {k:literal(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)): return [literal(x) for x in v]
    require(v is None or type(v) in (str,int,float,bool), 'literal renderer configuration')
    return v


def invariant(body):
    goal = source.vector(body.goal_state); pose = np.asarray(body.goal_pose)
    require(pose.shape == (3,) and np.isfinite(pose).all(), 'finite unchanged native goal_pose3')
    config = {k:literal(getattr(body,k)) for k in ('render_mode','render_size','window_size','render_action',
        'relative','action_scale','dt','control_hz')}
    config['variations'] = literal(body.variation_space.value)
    return dict(snapshot=moving.snapshot(body),goal_state=goal.tolist(),goal_state_float64_bits=goal.astype('<f8').tobytes().hex(),
        goal_pose=pose.tolist(),goal_pose_dtype=str(pose.dtype),goal_pose_bits=pose.tobytes().hex(),
        goal_rgb_sha256=moving.image_sha(body._goal),renderer=config)


def capture(env, obs, phase, step, mode, row, ops, store, index, old):
    """Persist available incident image/state before any acceptance requirement."""
    body = env.unwrapped; before = invariant(body)
    a = np.asarray(ops.call('explicit_render',env.render,phase=phase,step=step,mode=mode))
    receipt = dict(phase=phase,step=step,mode=mode,with_target=bool(body.with_target),
        before=before,shape=list(a.shape),dtype=str(a.dtype))
    row['last_observation'] = receipt
    valid = a.dtype == np.uint8 and a.shape == (224,224,3)
    if valid:
        a = a.copy(); name = f'b{index}_{phase}_{step:02}_{mode}.png'; h = store.frame(name,a); row['PNG'] += 1
        receipt.update(png=name,file_sha256=h,pixel_sha256=moving.image_sha(a))
    elif a.dtype.kind in 'buif':
        name = f'b{index}_{phase}_{step:02}_{mode}_invalid.npz'
        receipt.update(raw_render=name,file_sha256=store.arrays(name,raw_render=a))
    try:
        state = source.vector(body._get_obs()); after = invariant(body)
        receipt.update(state=state.tolist(),snapshot=after['snapshot'],observation_state=np.asarray(obs['state']).tolist(),after=after)
    except BaseException as exc:
        receipt['invariant_error']=dict(type=type(exc).__name__,message=str(exc))
        store.emit('fresh_observation',**ops.where,receipt=receipt)
        store.emit('guard_receipt',**ops.where,phase=phase,step=step,mode=mode,png=receipt.get('png'),accepted=False,
            reason='post-render invariant unavailable; persisted actual frame, not a stale observation')
        raise
    store.emit('fresh_observation',**ops.where,receipt=receipt)
    pixels = int(np.count_nonzero(np.all(a == [65,105,225],axis=-1))) if valid else 0
    matches = None
    if mode in ('off','off_after') and valid:
        matches = dict(file=receipt['file_sha256'] == old['file_sha256'],pixels=receipt['pixel_sha256'] == old['pixel_sha256'],
            state=moving.exact(state,np.asarray(old['state'],np.float64)),snapshot=receipt['snapshot'] == old['snapshot'],
            observation_state=moving.exact(np.asarray(obs['state']),np.asarray(old['observation_state'],np.float64)))
    flag_ok = body.with_target is (mode == 'on')
    accepted = (valid and pixels > 0 and bool(((state[:4]>=0)&(state[:4]<=512)).all()) and before == after and flag_ok and
        moving.exact(state,np.asarray(obs['state'])) and (matches is None or all(matches.values())))
    receipt['old_off_match'] = matches
    store.emit('guard_receipt',**ops.where,phase=phase,step=step,mode=mode,png=receipt.get('png'),accepted=accepted,
        canonical_agent_pixels=pixels,render_preserved_invariant=before == after,old_off_match=matches)
    require(accepted, 'fresh render/state/goal/configuration/old-off guard; incident retained before stop')
    if mode == 'off': row['off_verified'] += 1
    return a,receipt


def triplet(env, obs, step, first_image, first, row, ops, store, index, old):
    """First off is shared with the required old88 census; only TWO calls here."""
    body = env.unwrapped; require(body.with_target is False, 'off between dispatches')
    pair = dict(step=step,off_before=deepcopy(first),on=None,off_after=None,complete=False)
    row['checkpoints'].append(pair); fixed = first['after']
    try:
        body.with_target = True
        _,on = capture(env,obs,'plan',step,'on',row,ops,store,index,old); pair['on'] = deepcopy(on)
        require(on['before'] == on['after'] == fixed, 'on display must preserve same body, goal and renderer')
        body.with_target = False
        last_image,last = capture(env,obs,'plan',step,'off_after',row,ops,store,index,old); pair['off_after'] = deepcopy(last)
        require(last['before'] == last['after'] == fixed and moving.exact(last_image,first_image), 'off/on/off exact same body recovery')
        pair['complete'] = True; row['missing_checkpoints'].remove(step)
        store.emit('overlay_triplet_completed',**ops.where,step=step,off_before=first['png'],on=on['png'],off_after=last['png'],
            off_replay_exact=True,invariant_sha256=source.digest(fixed),on_semantics=contract()['on_semantics'])
    except BaseException as exc:
        low.retain(row,exc,'overlay'); raise
    finally:
        try:
            body.with_target = False
            store.emit('overlay_restored',**ops.where,step=step,with_target=bool(body.with_target))
        except BaseException as exc:
            low.retain(row,exc,'restore_overlay'); raise


@contextmanager
def native_hooks(env, ops, row):
    """Actual space.step/reset/setter hooks, not a count inferred from actions."""
    body = env.unwrapped; undo=[]; spaces=[]; resetting=False
    old_set,old_render,old_reset = body._set_state,body.render,env.reset
    def idle():
        for kind in ('goal_bind','explicit_render'):
            require(ops.active[kind] == 0, 'no hidden physics/setter/render during '+kind)
    def install():
        space=body.space
        if any(x[0] is space for x in spaces): return
        old=space.step; spaces.append((space,'step' in space.__dict__,old))
        def physics(*a,**kw):
            idle(); return ops.call('physics_step',lambda:old(*a,**kw))
        space.step=physics
    def setter(value):
        require(resetting and body.with_target is False, 'state setter only original target-free reset')
        install(); return ops.call('set_state',lambda:old_set(value))
    def render():
        require(ops.active['goal_bind'] == 0, 'no render during goal bind')
        return ops.call('goal_render',old_render) if resetting else old_render()
    def reset(*a,**kw):
        nonlocal resetting
        require(body.with_target is False, 'target-free reset only'); resetting=True
        try: return old_reset(*a,**kw)
        finally: resetting=False
    try:
        for obj,key,new in ((body,'_set_state',setter),(body,'render',render),(env,'reset',reset)):
            undo.append((obj,key,key in obj.__dict__,getattr(obj,key))); setattr(obj,key,new)
        yield
    except BaseException as exc: low.retain(row,exc,'branch')
    finally:
        for obj,owned,old in reversed(spaces):
            try:
                if owned: obj.step=old
                else: delattr(obj,'step')
            except BaseException as exc: low.retain(row,exc,'restore_space')
        for obj,key,owned,old in reversed(undo):
            try:
                if owned: setattr(obj,key,old)
                else: delattr(obj,key)
            except BaseException as exc: low.retain(row,exc,'restore_hook')


def branch(env, ref, source_plan, plan, saved, row, ops, store, index):
    """New moving replay kernel; NO old branch/execute/source episode calls."""
    body=env.unwrapped; slot=ref['slot']; warm=prior.raw_history(ref['raw_history']); actions=low.commands(plan['commands'])
    ops.where=dict(branch=index,root_slot=plan['root_slot'],arm=plan['arm'],selection_digest=plan['selection_digest'])
    store.emit('branch_started',**ops.where,plan=plan,raw_warmup=warm.tolist(),raw_float32_bits=warm.tobytes().hex())
    _,info=ops.call('reset',lambda:env.reset(seed=slot['reset_seed']),seed=slot['reset_seed'],kind_of_reset='original_seed')
    original_goal=source.vector(info['goal_state'])
    requested=source.sample_state(original_goal,list(body.block.shapes),dict(root_seed=slot['root_seed']),0,0,source_plan)
    store.emit('placement_request',**ops.where,original_goal=original_goal.tolist(),requested=requested.tolist(),placement_seed=0)
    obs,info=ops.call('reset',lambda:env.reset(seed=0,options=dict(state=requested.copy(),goal_state=original_goal.copy())),seed=0,kind_of_reset='placement')
    image,r=capture(env,obs,'warmup',0,'off',row,ops,store,index,saved['off'][('warmup',0)])
    frames=[image]; states=[source.vector(obs['state'])]; snaps=[r['snapshot']]
    for step,command in enumerate(warm,1):
        require(body.with_target is False and not body.eval_state(original_goal,source.vector(obs['state']))[0], 'native warmup terminal/off gate')
        obs,reward,terminated,truncated,info=ops.call('public_step',lambda c=command:env.step(c.copy()),phase='warmup',step=step,raw_action=command.tolist())
        row['warmup_returned']+=1
        store.emit('returned_transition',**ops.where,phase='warmup',step=step,command=command.tolist(),
            actual_command=np.asarray(body.latest_action).tolist(),actual_float32_bits=np.asarray(body.latest_action).tobytes().hex(),
            state=source.vector(obs['state']).tolist(),reward=float(reward),terminated=bool(terminated),truncated=bool(truncated))
        image,r=capture(env,obs,'warmup',step,'off',row,ops,store,index,saved['off'][('warmup',step)])
        require(moving.exact(np.asarray(body.latest_action),command), 'exact moving warmup raw action')
        require(not terminated and not truncated, 'warmup terminal; no plan execution')
        if step in (5,10): frames.append(image); states.append(source.vector(obs['state'])); snaps.append(r['snapshot'])
    require(moving.exact(np.asarray(frames),ref['history']) and moving.exact(np.asarray(states),ref['states']) and snaps == ref['dynamics'],
        'original moving blind RGB3/state/snapshot exact')
    goal,goal_rgb=ref['goal'],ref['goal_rgb']; root=source.vector(obs['state'])
    require(source.root_hash(root,goal) == ref['root_sha256'], 'exact moving root+goal identity')
    before=invariant(body); counts=deepcopy(ops.counts)
    ops.call('goal_bind',lambda:moving.attach_goal(body,goal,goal_rgb))
    after=invariant(body)
    require(after['snapshot'] == before['snapshot'] and after['goal_pose_bits'] == before['goal_pose_bits'] and
        after['renderer'] == before['renderer'] and body.with_target is False and
        all(v == ops.counts[k] for k,v in counts.items() if k != 'goal_bind') and
        moving.exact(np.asarray(body.goal_state),goal) and moving.exact(np.asarray(body._goal),goal_rgb), 'offline attachment only, no goal_pose edit or hidden operations')
    image,r=capture(env,obs,'plan',0,'off',row,ops,store,index,saved['off'][('plan',0)])
    row.update(initial_exact=True,root_state=root.tolist(),goal=goal.tolist(),goal_rgb_sha256=moving.image_sha(goal_rgb),
        goal_pose=after['goal_pose'],initial_metrics=low.metric(goal,root))
    require(not row['initial_metrics']['native_success'], 'old moving root must not be task terminal')
    store.emit('goal_attached',**ops.where,goal=goal.tolist(),goal_rgb_sha256=moving.image_sha(goal_rgb),goal_pose=after['goal_pose'])
    triplet(env,obs,0,image,r,row,ops,store,index,saved['off'][('plan',0)])
    for step,command in enumerate(actions,1):
        require(body.with_target is False, 'all control dispatches target-free')
        def dispatch(c=command): row['own_attempted']+=1; return env.step(c.copy())
        obs,reward,terminated,truncated,info=ops.call('public_step',dispatch,phase='plan',step=step,raw_action=command.tolist())
        row['own_returned']+=1; state=source.vector(obs['state']); metrics=low.metric(goal,state); success,distance=body.eval_state(goal,state)
        row.update(final_state=state.tolist(),final_metrics=metrics,native_success=bool(success),terminated=bool(terminated),truncated=bool(truncated))
        store.emit('returned_transition',**ops.where,phase='plan',step=step,command=command.tolist(),
            actual_command=np.asarray(body.latest_action).tolist(),actual_float32_bits=np.asarray(body.latest_action).tobytes().hex(),
            state=state.tolist(),reward=float(reward),terminated=bool(terminated),truncated=bool(truncated),
            returned_goal_state=np.asarray(info['goal_state']).tolist(),returned_goal_rgb_sha256=moving.image_sha(info['goal']))
        image,r=capture(env,obs,'plan',step,'off',row,ops,store,index,saved['off'][('plan',step)])
        require(moving.exact(np.asarray(body.latest_action),command), 'own selected raw action exact')
        require(moving.exact(np.asarray(info['goal_state']),goal) and moving.exact(np.asarray(info['goal']),goal_rgb), 'original task goal/off RGB retained')
        now=invariant(body)
        require({k:v for k,v in now.items() if k != 'snapshot'} == {k:v for k,v in after.items() if k != 'snapshot'}, 'goal/goal_pose/renderer unchanged across plan')
        require(bool(success) == metrics['native_success'] and float(distance) == metrics['native_distance'] and
            bool(terminated) == bool(success) and float(reward) == -float(distance), 'native reward/terminal contract unmodified')
        if step in (5,10): triplet(env,obs,step,image,r,row,ops,store,index,saved['off'][('plan',step)])
        if terminated or truncated:
            row['status']='native_success' if terminated and success and not truncated else 'native_stopped'; break
    else: row['status']='horizon_completed'
    row['full_horizon']=row['own_returned'] == 10
    row['endpoint_scope']='native_terminal' if row['terminated'] else 'full10' if row['full_horizon'] else 'native_early_stop'


def census(rows, counts):
    require(len(rows) == 4 and all(v['attempted'] == v['completed'] for v in counts.values()), 'fixed4 denominator, complete operation receipts')
    expected=dict(reset=8,set_state=16,goal_render=8,goal_bind=4,
        public_step=sum(r['warmup_returned']+r['own_returned'] for r in rows),
        explicit_render=sum(r['PNG'] for r in rows))
    expected['physics_step']=16+10*expected['public_step']
    require(counts == {k:dict(attempted=v,completed=v) for k,v in expected.items()}, 'actual internal operation census')
    require(all(r['off_verified'] == 12+r['own_returned'] and r['PNG'] == r['off_verified']+2*sum(c['complete'] for c in r['checkpoints'])
        for r in rows), 'original off + only two extra renders per completed triplet')


def execute(p, data, saved, store, factory, *, authorization):
    require(authorization == AUTH, 'explicit MAIN physics authorization')
    require(p['schema'] == NAME+'.prepare' and p['contract'] == contract() and len(saved) == len(p['source']['plans']) == 4 and
        [(x['root_slot'],x['arm']) for x in p['source']['plans']] == [(i,a) for i in range(2) for a in ARMS], 'new fixed four-arm contract')
    rows=[dict(root_slot=x['root_slot'],arm=x['arm'],selection_digest=x['selection_digest'],status='not_started',first_error=None,secondary_errors=[],
        warmup_returned=0,own_attempted=0,own_returned=0,initial_exact=False,full_horizon=False,native_success=False,terminated=False,truncated=False,
        PNG=0,off_verified=0,checkpoints=[],missing_checkpoints=[0,5,10]) for x in p['source']['plans']]
    result=dict(schema=NAME+'.report',status='running',prepare_id=source.digest(p),rows=rows,errors=[],contract=contract(),fixed_denominator=4,
        original_report_sha256=REPORT_SHA,sourcepack_sha256=SOURCEPACK_SHA,global_freshness='known_moving_roots_calibration_only',
        new_independent_samples=0,new_neural_calls=0,training=False,P1_admission=False,source_goal_on_certified=False)
    result['incident_scope']='last_observation is last completed explicit render; internal step exceptions do not guarantee a post-fault frame'
    ops=Ops(p,store)
    def error(stage,exc): result['errors'].append(dict(stage=stage,type=type(exc).__name__,message=str(exc)))
    try:
        store.emit('run_started',prepare=p)
        for i,(plan,row,old) in enumerate(zip(p['source']['plans'],rows,saved,strict=True)):
            env=None
            try:
                store.check(FINAL); env=factory(); moving.validate_env(env.unwrapped)
                require(env.unwrapped.render_action is False and env.unwrapped.render_size == 224 and env.unwrapped.render_mode == 'rgb_array', 'native off RGB224/no action marker')
                row['status']='running'
                with native_hooks(env,ops,row): branch(env,data['references'][plan['root_slot']],data['plan'],plan,old,row,ops,store,i)
            except BaseException as exc: low.retain(row,exc,'setup_or_branch')
            finally:
                if env is not None:
                    try: env.close()
                    except BaseException as exc: low.retain(row,exc,'close')
                store.emit('branch_closed',receipt=row)
    except BaseException as exc: error('execution',exc)
    result['inputs_final_verified']=False
    try:
        for path,h in p['bindings'].items(): require(sha(path) == h, 'final input/source seal '+path)
        result['inputs_final_verified']=True
    except BaseException as exc: error('final_bindings',exc)
    clean=not result['errors'] and all(r['first_error'] is None and r['status'] in ('horizon_completed','native_success','native_stopped') for r in rows)
    if clean:
        try: census(rows,ops.counts)
        except BaseException as exc: error('final_census',exc); clean=False
    result.update(status='moving_target_overlay_plans_recorded' if clean else 'partial_or_failed',counts=ops.counts,
        full_horizon_count=sum(r['full_horizon'] for r in rows),native_successes=sum(r['native_success'] for r in rows),
        complete_triplets=sum(c['complete'] for r in rows for c in r['checkpoints']),wall_seconds=time.monotonic()-ops.started)
    try: store.emit('run_finished',status=result['status'],counts=ops.counts,errors=result['errors'],fixed_denominator=4)
    except BaseException as exc: error('final_journal',exc); result['status']='partial_or_failed'
    try: result['journal_sha256']=sha(store.path/'journal.jsonl'); result['journal_chain_tail']=store.previous
    except BaseException as exc: error('final_journal_hash',exc); result['status']='partial_or_failed'
    try: store.put('report.json',result,final=True)
    except BaseException as exc: error('final_report',exc); result['status']='partial_or_failed'
    return result


def run(prepared, expected, output, factory, *, authorization):
    require(authorization == AUTH, 'MAIN authorization before reads/factory')
    p=verify(prepared,expected); value,data,saved=inputs(); require(value == p['source'], 'input read closure unchanged')
    store=Store(output); store.put('execution_binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected))
    return execute(p,data,saved,store,factory,authorization=authorization)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('command',choices=('prepare','verify','run'))
    for key in ('prepared','prepare-sha','output'): parser.add_argument('--'+key)
    parser.add_argument('--authorize-physics',action='store_true'); a=parser.parse_args()
    if a.command == 'prepare': r=prepare(a.output)
    elif a.command == 'verify': r=dict(schema=verify(a.prepared,a.prepare_sha)['schema'])
    else:
        require(a.authorize_physics, 'MAIN only; no physics before complete review')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False,render_action=False)
        print('Fixed known moving4 arms, display-only pairs; no selection or P1 admission.',flush=True)
        r=run(a.prepared,a.prepare_sha,a.output,factory,authorization=AUTH)
        print(source.canonical({k:v for k,v in r.items() if k != 'rows'}))
        if r['status'] != 'moving_target_overlay_plans_recorded': raise SystemExit(2)
        return
    print(source.canonical(r))


if __name__ == '__main__': main()
