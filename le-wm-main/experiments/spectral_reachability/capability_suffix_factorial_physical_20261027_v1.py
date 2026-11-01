"""MAIN-only deterministic prefix replay plus 24 NEW v2 model-only suffixes.

v1 CUDA plans are never accepted. Saved-only preparation requires external
v2 producer/consumer source SHAs AND MAIN's verified24 saved acceptance SHA.
No neural runner, approximate state restore, retries or scientific admission.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import moving_overlay_capability_probe_physical_20261027_v1 as probe

NAME = 'capability_suffix_factorial_physical_20261027_v1'
PRODUCER = 'capability_suffix_factorial_cuda_MAIN_20261027_v2'
CONSUMER = 'consume_suffix_factorial_MAIN_20261027_v2'
AUTH = 'MAIN_FACTORIAL24_V2_SUFFIX_PHYSICS_ONLY'
VARIANTS = ('open_full', 'teacher_full', 'teacher_small')
base, ROOT = probe.base, probe.ROOT
require, sha, raw, read = probe.require, probe.sha, probe.raw, probe.read
CAP, FLOOR, FINAL, PREPARE_CAP = 192*1024**2, 200*1024**2, 8*1024**2, 16*1024**2
PROBE_SHA = 'fed070fdca9445019fb2643e6e507a2f0a991a3a14439df88c46f56a65ae923b'
BASE_SHA = 'd6ee57d4d902c2d8c865c8372718a3bbbea3964a23973da535f106e415229570'


def contract():
    return dict(anchors=8, variants=list(VARIANTS), fixed_denominator=24, independent_roots=2,
        warmup=10, prefix=5, suffix=5, raw_dtype='float32', state_dimension=7,
        action_scale=100, relative=True, raw_action_bounds=[-1, 1], no_goal_pose_edit=True,
        control_with_target=False, flags=[False, True, False], wall_seconds=900,
        output_cap_bytes=CAP, disk_floor_bytes=FLOOR, final_reserve_bytes=FINAL,
        restoration='fresh reset + original warm10 + original prefix5 + all51 exact off/on/off observations; no suffix setter',
        cache_identity='deterministic reconstruction, NOT serialized/cache-bit certification',
        caps=dict(constructor=24, close=24, reset=48, public_step=480, physics_step=4896,
            set_state=96, goal_setter=72, goal_render=48, goal_bind=24, explicit_render=1584),
        producer=PRODUCER, consumer=CONSUMER, CUDA_v1_accepted=False,
        suffix_source='own sealed v2 CUDA raw buffer through MAIN verified24 consumer; no NumPy tanh reconstruction',
        count_scope='dispatch intent separate from actual callback entry/return; failure never refunds',
        no_neural=True, training=False, Qfit=False, P1_admission=False, scientific_admission=False,
        independent_confirmation=False, equal_total_compute=False, equal_FLOPs=False,
        interpretation='exploratory known-root context/action-scale ablation; no scientific pass or winning claim')


def absolute(value):
    require(type(value) is str and Path(value).is_absolute() and str(Path(value).resolve()) == value,
            'canonical absolute nonsymlink path required')
    return value


def raw5(value):
    a = np.asarray(value)
    require(a.shape == (5, 2) and a.dtype.kind in 'fi' and np.isfinite(a).all() and
            np.max(np.abs(a)) <= 1 and np.array_equal(a, a.astype(np.float32)), 'literal lossless float32 raw5x2')
    return a.astype(np.float32)


def key(row): return row['root_slot'], row['condition'], row['arm']


def merge(bindings, additions):
    for f, h in additions.items():
        absolute(f); base.hash_value(h)
        require(f not in bindings or bindings[f] == h, 'nonconflicting complete input seals')
        bindings[f] = h


def own_files():
    return [Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md'),
            ROOT.parent/'docs/D2P_CAPABILITY_SUFFIX_FACTORIAL_PROTOCOL_MAIN_20261027.md',
            ROOT.parent/'docs/D2P_CAPABILITY_SUFFIX_FACTORIAL_PROTOCOL_MAIN_20261027_v2.md', *probe.own_files()]


def validate_verified(compiler, prepared, verified, originals, compiler_sha):
    """Real producer layout RAM seam; consume is mandatory, raw release is false."""
    require(compiler['schema'] == PRODUCER+'.report' and compiler['status'] == 'all24_plans_sealed' and
            compiler['errors'] == [] and compiler['commands_released'] is False and compiler['P1_admission'] is False,
            'v2 raw compiler only; v1 NEVER releasable by this adapter')
    require(prepared['schema'] == PRODUCER+'.prepare' and prepared['contract'] == compiler['contract'] and
            len(prepared['cases']) == len(compiler['rows']) == 24 and len(originals) == 8, 'fixed v2 case layout')
    forwards = compiler['forwards']
    require(compiler['predict_API'] == len(forwards) == 312 and
            all(type(f['index']) is int and f['index'] == i and type(f['case']) is int and f['case'] == i//13 and
                f['completed'] is True for i, f in enumerate(forwards)), 'all312 actual v2 CUDA raw forward identities')
    require(verified['schema'] == CONSUMER+'.report' and verified['status'] == 'all24_saved_verified' and
            verified['source_report_sha256'] == compiler_sha and verified['prepare_sha256'] == compiler['prepare_sha256'] and
            verified['branches'] == 24 and verified['known_roots'] == 2 and verified['fresh_roots'] == 0 and
            verified['commands_released'] is True and verified['P1_admission'] is False and
            verified['new_neural_calls'] == verified['new_physics_calls'] == 0 and len(verified['plans']) == 24,
            'MAIN v2 saved verified24 required, not raw commands_released flag')
    selected = []
    for i, (case, row, plan) in enumerate(zip(prepared['cases'], compiler['rows'], verified['plans'], strict=True)):
        anchor = i//3; variant = VARIANTS[i%3]; original = originals[anchor]
        require(type(case['index']) is type(row['index']) is type(plan['index']) is int and
                case['index'] == row['index'] == plan['index'] == i and type(case['anchor']) is type(plan['anchor']) is int and
                case['anchor'] == plan['anchor'] == anchor and case['variant'] == plan['variant'] == variant and
                case['scale'] == plan['scale'] == (.1 if variant == 'teacher_small' else 1.) and row['status'] == 'sealed',
                'all24 original case order/variant/scale, not outcome selection')
        require(type(case['root_slot']) is type(plan['root_slot']) is int and key(case) == key(plan) == key(original) and
                all(case[k] == plan[k] == original[k] for k in ('task_id', 'selection_digest')),
                'exact original root/condition/arm/task/prefix selection binding')
        require(case['original_plan'] == plan['original_plan'] and
                all(case['original_plan'][k] == original[k] for k in base.PLAN_KEYS), 'exact original plan, not v1/new prefix')
        prefix = base.previous.commands(original['commands'])[:5]
        require(raw5(case['prefix']).tobytes() == raw5(plan['prefix']).tobytes() == prefix.tobytes(), 'original prefix5 bits unchanged')
        winner = row['result']['winner']
        require(isinstance(winner.get('raw_receipt'), dict) and bool(winner['raw_receipt']), 'v2 actual CUDA raw receipt mandatory')
        require(type(winner['index']) is int and 0 <= winner['index'] < 10 and
                plan['raw_receipt'] == winner['raw_receipt'], 'plan actual CUDA raw receipt equals own winner')
        suffix = raw5(winner['raw'])
        selected_forward = forwards[13*i+2+winner['index']]; replay_forward = forwards[13*i+12]
        require(selected_forward['raw_receipt'] == winner['raw_receipt'] and
                raw5(selected_forward['raw']).tobytes() == raw5(replay_forward['raw']).tobytes() == suffix.tobytes() and
                replay_forward['raw_receipt'] == row['result']['replay_raw_receipt'], 'winner to actual forward+independent raw replay exact bits')
        require(raw5(plan['suffix']).tobytes() == suffix.tobytes() and
                np.max(np.abs(suffix)) <= case['scale'] and plan['commands'] == plan['prefix']+plan['suffix'] and
                plan['winner_index'] == winner['index'] and plan['predicted_Q'] == winner['Q'], 'own v2 sealed nonlinear winner raw buffer')
        # Deliberately NEVER np.tanh(params): CUDA and NumPy differ by ULP.
        seal = base.source.digest(dict(compiler_report_sha256=compiler_sha, case=case, winner=winner))
        item = dict(plan=deepcopy(plan), suffix_selection_sha256=seal,
            case_sha256=base.source.digest(case), winner_raw_receipt=deepcopy(winner['raw_receipt']))
        item['execution_sha256'] = base.source.digest(item); selected.append(item)
    return selected


def compiler_material(arguments):
    """External actual hashes before import; exact saved MAIN acceptance required."""
    output = Path(absolute(arguments['compiler_output']))
    rp = output/'report.json'; expected = arguments['compiler_report_sha256']
    for k in ('compiler_report_sha256', 'producer_source_sha256', 'consumer_source_sha256', 'consumer_report_sha256'):
        base.hash_value(arguments[k])
    producer_path = ROOT/'experiments/spectral_reachability'/(PRODUCER+'.py')
    consumer_path = ROOT/'experiments/spectral_reachability'/(CONSUMER+'.py')
    require(sha(producer_path) == arguments['producer_source_sha256'] and
            sha(consumer_path) == arguments['consumer_source_sha256'], 'MAIN FINAL v2 producer/consumer source SHA BEFORE import')
    acceptance_path = Path(absolute(arguments['consumer_report']))
    require(sha(rp) == expected and sha(acceptance_path) == arguments['consumer_report_sha256'], 'external actual compiler AND MAIN acceptance hashes')
    saved = read(acceptance_path, PREPARE_CAP)
    require(saved['schema'] == CONSUMER+'.report' and saved['status'] == 'all24_saved_verified' and
            saved['source_report_sha256'] == expected, 'v2 acceptance, never v1 or raw release shortcut')
    consumer = importlib.import_module('experiments.spectral_reachability.'+CONSUMER)
    require(Path(consumer.__file__).resolve() == consumer_path and sha(consumer_path) == arguments['consumer_source_sha256'], 'fixed v2 consumer identity')
    verified = consumer.consume(output, expected)
    require(verified == saved, 'MAIN saved acceptance exactly equals v2 own-consume, including all24 winners')
    report = read(rp, 64*1024**2); prepared_path = Path(absolute(report['prepared']))
    require(sha(prepared_path) == report['prepare_sha256'], 'compiler actual prepare SHA')
    prepared = read(prepared_path, PREPARE_CAP)
    bindings = {}
    merge(bindings, prepared['bindings'])
    merge(bindings, {str(rp): expected, str(prepared_path): report['prepare_sha256'],
        str(producer_path): arguments['producer_source_sha256'], str(consumer_path): arguments['consumer_source_sha256'],
        str(acceptance_path): arguments['consumer_report_sha256']})
    for name, h in report['artifact_sha256'].items():
        require(Path(name).name == name and name not in ('', '.', '..'), 'flat v2 producer artifact')
        merge(bindings, {str(output/name): h})
    return report, prepared, verified, bindings


def constructor_contract():
    path = base.moving.ENV
    require(sha(path) == base.moving.ENV_SHA, 'sealed native constructor source')
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PushT')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    require(not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in
                    {'reset', 'step', 'render', '_set_state', '_set_goal_state', '_setup'} for n in ast.walk(init)),
            'native constructor inert BEFORE hooks; partial failures still unknown')
    return str(Path(path).resolve()), base.moving.ENV_SHA


def material(arguments):
    required = {'compiler_output', 'compiler_report_sha256', 'producer_source_sha256', 'consumer_source_sha256',
                'consumer_report', 'consumer_report_sha256', 'physical_output', 'physical_report_sha256'}
    require(type(arguments) is dict and set(arguments) == required, 'explicit actual v2 compiler/consumer + original8 physical arguments')
    require(sha(probe.__file__) == PROBE_SHA and sha(base.__file__) == BASE_SHA, 'sealed deterministic replay primitives')
    report, compiled, verified, bindings = compiler_material(arguments)
    # SAVED validation only: never runs old40 probes or old8 physical execution.
    original, data = probe.material(absolute(arguments['physical_output']), arguments['physical_report_sha256'])
    selected = validate_verified(report, compiled, verified, original['plans'], arguments['compiler_report_sha256'])
    merge(bindings, original['bindings'])
    native, h = constructor_contract(); merge(bindings, {native: h})
    for f in own_files(): merge(bindings, {str(f.resolve()): sha(f)})
    for f, h in bindings.items(): require(sha(f) == h, 'final saved source/compiler/task/raw closure')
    p = dict(schema=NAME+'.prepare', arguments=deepcopy(arguments), contract=contract(),
        plans=deepcopy(original['plans']), anchors=deepcopy(original['anchors']), suffix_plans=selected,
        physical_report_sha256=arguments['physical_report_sha256'], planner_report_sha256=original['planner_report_sha256'],
        compiler_report_sha256=arguments['compiler_report_sha256'], consumer_report_sha256=arguments['consumer_report_sha256'],
        MAIN_verified24=True, bindings=bindings)
    return p, data


def prepare(output, arguments):
    p, _ = material(arguments); payload = raw(p); path = Path(output).absolute()
    require(len(payload) <= PREPARE_CAP and not path.exists() and not path.is_symlink() and
            shutil.disk_usage(path.parent).free >= FLOOR+len(payload), 'exclusive bounded prepare +200MiB floor')
    with path.open('xb') as f:
        require(f.write(payload) == len(payload), 'short prepare'); f.flush(); os.fsync(f.fileno())
    base.paired.fsync_dir(path.parent)
    return dict(prepared=str(path), prepare_sha256=sha(path), bytes=len(payload))


def verify(prepared, expected):
    base.hash_value(expected); require(sha(prepared) == expected, 'external physical prepare SHA')
    p = read(prepared, PREPARE_CAP); now, data = material(p['arguments'])
    require(p == now and sha(prepared) == expected, 'full v2 verified24/original51trace/selected raw prepare reconstruction')
    return p, data


class Store(base.Store):
    def __init__(self, path):
        self.path = Path(path).absolute()
        require(not self.path.exists() and not self.path.is_symlink() and
                shutil.disk_usage(self.path.parent).free >= FLOOR+CAP, 'NEW192MiB +200MiB floor')
        self.path.mkdir(); base.paired.fsync_dir(self.path.parent); self.previous = None; self.index = 0

    def check(self, size=4096, final=False):
        files = list(self.path.iterdir())
        require(all(f.is_file() and not f.is_symlink() for f in files), 'flat regular output')
        require(sum(f.stat().st_size for f in files)+size <= CAP-(0 if final else FINAL) and
                shutil.disk_usage(self.path).free >= FLOOR+size, '192MiB including partial/final +200MiB floor')


class Meter:
    """Own exact caps; callback entry/return distinct from durable intent."""
    def __init__(self, store, started=None):
        self.store = store; self.started = time.monotonic() if started is None else started
        self.where = {}; self.secondary = []; self.fault = False; self.receipts = []
        self.counts = {k: dict(attempted=0, completed=0) for k in contract()['caps']}
        self.active = {k: 0 for k in self.counts}

    def call(self, kind, fn, *, cleanup=False, **fields):
        require(not cleanup or kind == 'close', 'cleanup only close')
        if not cleanup:
            require(not self.fault and time.monotonic()-self.started < 900, '900s/fault predispatch closure')
            self.store.check()
        c = self.counts[kind]; require(c['attempted'] < contract()['caps'][kind], 'predispatch '+kind+' cap')
        r = dict(index=len(self.receipts), resource=kind, **deepcopy(self.where), **deepcopy(fields), status='pending')
        try: self.store.emit('operation_dispatch_intent', receipt=r)
        except BaseException as exc:
            self.fault = True
            if not cleanup: raise
            self.secondary.append(base.error('cleanup_intent', exc))
        self.receipts.append(r); c['attempted'] += 1; self.active[kind] += 1
        try:
            value = fn(); c['completed'] += 1; r['status'] = 'returned'
        except BaseException as exc:
            r.update(status='failed', error=base.error(kind, exc))
            try: self.store.emit('operation_failed', receipt=r)
            except BaseException as secondary:
                self.fault = True; self.secondary.append(base.error('failed_journal', secondary))
            raise
        finally: self.active[kind] -= 1
        try: self.store.emit('operation_returned', receipt=r)
        except BaseException as exc:
            self.fault = True
            if not cleanup: raise
            self.secondary.append(base.error('cleanup_return', exc))
        return value


def episode(env, root, source_plan, original, anchor, selected, row, meter, store):
    """Fresh actual recipe; no setter/restoration after original prefix5."""
    store = probe.TraceStore(store); body = env.unwrapped; packet = root['packets'][original['condition']]
    obs, native_goal = base.place(env, source_plan, root['slot'], meter, store)
    frames = [base.capture(env, obs, 'warmup', 0, row, meter, store, root['warm'][0])]
    for step, command in enumerate(base.moving.warmup_commands(), 1):
        require(not body.eval_state(native_goal, base.source.vector(obs['state']))[0], 'warmup already terminal')
        obs, info, t, tr, reward, actual = base.dispatch(env, command, 'warmup', step, row, meter, store, native_goal)
        image = base.capture(env, obs, 'warmup', step, row, meter, store, root['warm'][step])
        base.paired.check_return(env, command, actual, info, native_goal, reward, t)
        require(not t and not tr, 'warmup stop, no padded history')
        if step in (5, 10): frames.append(image)
    require(base.moving.exact(np.asarray(frames), packet['rgb_history']) and
            base.moving.exact(packet['raw_action_history'], base.moving.warmup_commands()), 'exact source history3/raw10')
    root_state = base.source.vector(obs['state']); goal = base.source.vector(root['goal'])
    require(base.source.root_hash(root_state, goal) == root['root_sha256'], 'original moving root+goal')
    base.bind_goal(body, goal, packet['goal_rgb'], meter, store); bound = base.invariant(body)
    row.update(root_state=root_state.tolist(), goal=goal.tolist(), root_sha256=root['root_sha256'],
        goal_pixel_sha256=base.moving.image_sha(packet['goal_rgb']), goal_pose=bound['goal_pose'])
    base.capture(env, obs, 'plan', 0, row, meter, store)
    require(not body.eval_state(goal, root_state)[0], 'root already terminal')
    for step, command in enumerate(base.previous.commands(original['commands'])[:5], 1):
        obs, info, t, tr, reward, actual = base.dispatch(env, command, 'plan', step, row, meter, store, goal)
        base.capture(env, obs, 'plan', step, row, meter, store)
        base.paired.check_return(env, command, actual, info, goal, reward, t)
        require(not t and not tr, 'original prefix stop; no suffix approximation')
    probe.compare_trace(row, anchor, store, meter)  # ALL51, not only chosen PNG/native7
    require({k: v for k, v in base.invariant(body).items() if k != 'snapshot'} ==
            {k: v for k, v in bound.items() if k != 'snapshot'}, 'original goal/renderer unchanged')
    start = base.source.vector(obs['state']); row.update(start_verified=True, start_state=start.tolist())
    for step, command in enumerate(raw5(selected['plan']['suffix']), 1):
        obs, info, t, tr, reward, actual = base.dispatch(env, command, 'suffix', step, row, meter, store, goal)
        base.capture(env, obs, 'suffix', step, row, meter, store)
        success = base.paired.check_return(env, command, actual, info, goal, reward, t)
        now = base.invariant(body)
        require({k: v for k, v in now.items() if k != 'snapshot'} == {k: v for k, v in bound.items() if k != 'snapshot'} and
                base.moving.exact(np.asarray(info['goal']), packet['goal_rgb']), 'goal/goal_pose/renderer invariant')
        state = base.source.vector(obs['state']); delta = state-start
        metrics = base.previous.metric(goal, state)
        response = dict(step=step, state=state.tolist(), delta_native7=delta.tolist(),
            delta_angle_wrapped=float(np.arctan2(np.sin(delta[4]), np.cos(delta[4]))),
            reward=reward, native_success=bool(success), terminated=t, truncated=tr, native_metrics=metrics)
        row['responses'].append(response); row.update(final_state=state.tolist(), native_success=bool(success),
            terminated=t, truncated=tr, final_metrics=metrics)
        if step == 5:
            last = row['trajectory'][-1]
            row['checkpoints'].append(dict(step=10, phase='suffix', suffix_step=5,
                **{k: deepcopy(last[k]) for k in ('png', 'file_sha256', 'pixel_sha256', 'state', 'snapshot')}))
            row['missing_checkpoints'].remove(10)
        store.emit('suffix_response', **meter.where, response=response)
        if t or tr: row['status'] = 'native_stopped'; return
    row['status'] = 'suffix_completed'


def validate_execution(p):
    require(p['schema'] == NAME+'.prepare' and p['contract'] == contract() and p['MAIN_verified24'] is True and
            len(p['plans']) == len(p['anchors']) == 8 and len(p['suffix_plans']) == 24, 'own frozen v2 verified24 prepare')
    expected = [(s, c, a) for s in (0, 1) for c in base.CONDITIONS for a in base.ARMS]
    require([key(x) for x in p['plans']] == expected, 'all8 unchanged prefix anchors')
    for i, x in enumerate(p['suffix_plans']):
        plan = x['plan']; original = p['plans'][i//3]
        require(type(plan['index']) is type(plan['anchor']) is int and plan['index'] == i and plan['anchor'] == i//3 and
                plan['variant'] == VARIANTS[i%3] and key(plan) == key(original) and
                all(plan[k] == original[k] for k in ('task_id', 'selection_digest')) and
                all(plan['original_plan'][k] == original[k] for k in base.PLAN_KEYS), 'full24 v2 suffix identity')
        require(raw5(plan['prefix']).tobytes() == base.previous.commands(original['commands'])[:5].tobytes(), 'fixed original prefix bits')
        raw5(plan['suffix']); base.hash_value(x['suffix_selection_sha256']); base.hash_value(x['case_sha256'])
        require(x['execution_sha256'] == base.source.digest({k: v for k, v in x.items() if k != 'execution_sha256'}) and
                plan['raw_receipt'] == x['winner_raw_receipt'], 'immutable raw receipt/selected suffix execution seal')


def census(rows, counts):
    c = {k: v['completed'] for k, v in counts.items()}
    require(all(v['attempted'] == v['completed'] for v in counts.values()) and
        c['constructor'] == c['close'] == 24 and c['reset'] == 48 and c['goal_bind'] == 24 and
        c['physics_step'] == 2*c['reset']+10*c['public_step'] and c['set_state'] == 2*c['reset'] and
        c['goal_render'] == c['reset'] and c['goal_setter'] == c['reset']+c['goal_bind'] and
        c['public_step'] == sum(r['warmup_returned']+r['prefix_returned']+r['suffix_returned'] for r in rows) and
        c['explicit_render'] == sum(r['PNG'] for r in rows), 'actual full/earlystop resource census, not padded caps')


def execute(p, data, store, factory, *, authorization, prepared, prepare_sha256, started=None):
    require(authorization == AUTH, 'MAIN explicit physics authorization BEFORE factory')
    base.hash_value(prepare_sha256); validate_execution(p)
    rows = [dict(index=i, anchor_index=i//3, **{k: x['plan'][k] for k in ('root_slot', 'condition', 'arm', 'task_id', 'selection_digest', 'variant', 'scale')},
        selected_suffix=deepcopy(x), status='not_started', first_error=None, secondary_errors=[],
        start_verified=False, start_state=None, final_state=None, final_metrics=None, native_success=None,
        terminated=None, truncated=None, constructor_internal_work='not_attempted', responses=[], trajectory=[],
        checkpoints=[], missing_checkpoints=[0, 5, 10], PNG=0, warmup_attempted=0, warmup_returned=0, own_attempted=0, own_returned=0,
        prefix_attempted=0, prefix_returned=0, suffix_attempted=0, suffix_returned=0)
        for i, x in enumerate(p['suffix_plans'])]
    meter = Meter(store, started); errors = []; seen = []; final_ok = False
    binding = dict(prepared=str(Path(prepared).absolute()), prepare_sha256=prepare_sha256,
        compiler_report_sha256=p['compiler_report_sha256'], consumer_report_sha256=p['consumer_report_sha256'],
        physical_report_sha256=p['physical_report_sha256'])
    def fail(stage, exc, row=None):
        e = base.error(stage, exc); errors.append(e)
        if row is not None:
            if row['first_error'] is None: row['first_error'] = deepcopy(e)
            else: row['secondary_errors'].append(deepcopy(e))
            row['status'] = 'engineering_failed'
    try:
        store.put('execution_binding.json', binding); store.emit('run_started', prepare_sha256=prepare_sha256, contract=contract())
        for row in rows:
            i = row['index']; meter.where = dict(branch=i, anchor_index=i//3, **{k: row[k] for k in
                ('root_slot', 'condition', 'arm', 'variant', 'selection_digest')})
            env = None; holder = []
            def construct():
                row['constructor_internal_work'] = 'unknown'
                value = factory(); holder.append(value); return value
            try:
                env = meter.call('constructor', construct)
                require(all(env is not x and env.unwrapped is not x.unwrapped for x in seen), 'fresh env for EACH suffix')
                seen.append(env); base.validate_env(env.unwrapped)
                require(all(getattr(env.unwrapped, k, None) is None for k in ('space', 'latest_action', 'render_buffer')), 'pristine constructor before hooks')
                require(env.unwrapped.render_action is False and env.unwrapped.render_mode == 'rgb_array', 'native renderer')
                row.update(status='running', constructor_internal_work='sealed_native_pristine_no_physics')
                with base.native_hooks(env, meter, row['secondary_errors']):
                    episode(env, data['roots'][row['root_slot']], data['source_plan'], p['plans'][i//3],
                            p['anchors'][i//3], p['suffix_plans'][i], row, meter, store)
                require(not row['secondary_errors'], 'hook cleanup failed')
            except BaseException as exc: fail('branch', exc, row)
            finally:
                if env is None and holder: env = holder[0]
                if env is not None:
                    try: meter.call('close', env.close, cleanup=True)
                    except BaseException as exc: fail('close', exc, row)
                steps = [x for x in meter.receipts if x.get('branch') == i and x['resource'] == 'public_step']
                for phase in ('warmup', 'plan', 'suffix'):
                    prefix = 'prefix' if phase == 'plan' else phase
                    row[prefix+'_attempted'] = sum(x.get('phase') == phase for x in steps)
                    row[prefix+'_returned'] = sum(x.get('phase') == phase and x['status'] == 'returned' for x in steps)
                row['own_attempted'] = row['prefix_attempted']+row['suffix_attempted']
                row['own_returned'] = row['prefix_returned']+row['suffix_returned']
                row['full_suffix'] = row['suffix_returned'] == 5
                if row['secondary_errors'] and row['first_error'] is None: fail('cleanup', ValueError('see secondary_errors'), row)
            store.emit('branch_closed', branch=i, receipt=row)
    except BaseException as exc: fail('execution', exc)
    try:
        for path, h in p['bindings'].items(): require(sha(path) == h, 'final input/source seal')
        final_ok = True
    except BaseException as exc: fail('final_inputs', exc)
    errors.extend(meter.secondary)
    if time.monotonic()-meter.started >= 900: fail('wall', ValueError('900s exhausted; cleanup/finalization retained'))
    if not errors:
        try: census(rows, meter.counts)
        except BaseException as exc: fail('census', exc)
    good = not errors and final_ok and all(r['status'] in ('suffix_completed', 'native_stopped') for r in rows)
    status = 'factorial24_suffix_physical_recorded_not_admitted' if good else 'partial_or_failed'
    report = dict(schema=NAME+'.report', status=status, rows=rows, errors=errors, fixed_denominator=24,
        independent_roots=2, execution_binding=binding, contract=contract(), counts=meter.counts,
        actual_operation_receipts=meter.receipts, inputs_final_verified=final_ok,
        full_suffixes=sum(r.get('full_suffix', False) for r in rows),
        early_stops=sum(r['status'] == 'native_stopped' for r in rows), engineering_failures=sum(r['status'] == 'engineering_failed' for r in rows),
        not_started=sum(r['status'] == 'not_started' for r in rows),
        total_internal_physics_scope='unknown_constructor_work' if any(r['constructor_internal_work'] == 'unknown' for r in rows) else 'hook_observed_native_work',
        new_neural_calls=0, new_independent_samples=0, training=False, P1_admission=False, scientific_admission=False,
        independent_confirmation=False, equal_total_compute=False, equal_FLOPs=False, wall_seconds=time.monotonic()-meter.started)
    try:
        store.emit('run_finished', status=status, counts=meter.counts, errors=errors)
        report.update(journal_sha256=sha(store.path/'journal.jsonl'), journal_chain_tail=store.previous,
                      execution_binding_sha256=sha(store.path/'execution_binding.json'))
    except BaseException as exc: fail('final_journal', exc); report['status'] = 'partial_or_failed'
    try: store.put('report.json', report, final=True)
    except BaseException as exc: fail('final_report', exc); report['status'] = 'partial_or_failed'
    return report


def run(prepared, expected, output, factory, *, authorization):
    require(authorization == AUTH, 'MAIN authorization BEFORE verify/factory')
    started = time.monotonic(); p, data = verify(prepared, expected)
    return execute(p, data, Store(output), factory, authorization=authorization,
                   prepared=prepared, prepare_sha256=expected, started=started)


def validate_saved(report, events, p):
    """Pure real-report-layout RAM seam; damaged/incomplete journals fail closed."""
    validate_execution(p)
    require(report['schema'] == NAME+'.report' and report['status'] in
            ('factorial24_suffix_physical_recorded_not_admitted', 'partial_or_failed') and
            report['fixed_denominator'] == len(report['rows']) == 24 and report['contract'] == contract() and
            report['execution_binding']['compiler_report_sha256'] == p['compiler_report_sha256'] and
            report['execution_binding']['consumer_report_sha256'] == p['consumer_report_sha256'], 'own24 physical report and v2 acceptance identity')
    for flag in ('training', 'P1_admission', 'scientific_admission', 'independent_confirmation', 'equal_total_compute', 'equal_FLOPs'):
        require(report[flag] is False, 'no science/compute upgrade')
    require(report['new_neural_calls'] == report['new_independent_samples'] == 0 and
            events[0]['kind'] == 'run_started' and events[0]['prepare_sha256'] == report['execution_binding']['prepare_sha256'] and
            events[-1]['kind'] == 'run_finished' and events[-1]['status'] == report['status'] and
            events[-1]['counts'] == report['counts'] and events[-1]['errors'] == report['errors'], 'complete terminal physical journal')
    intents = {}; terminals = {}; stack = []
    for e in events:
        if e['kind'] not in ('operation_dispatch_intent', 'operation_returned', 'operation_failed'): continue
        r = e['receipt']; i = r['index']
        require(type(i) is int and r['resource'] in contract()['caps'], 'literal actual resource/index')
        if e['kind'] == 'operation_dispatch_intent':
            require(i == len(intents) and r['status'] == 'pending', 'sequential dispatch intents')
            intents[i] = r; stack.append(i)
        else:
            require(i in intents and i not in terminals and stack and stack.pop() == i and
                    r['status'] == ('returned' if e['kind'] == 'operation_returned' else 'failed') and
                    dict({k: v for k, v in r.items() if k != 'error'}, status='pending') == intents[i], 'actual nested entry/return identity')
            terminals[i] = r
    require(not stack and len(intents) == len(terminals) and
            report['actual_operation_receipts'] == [terminals[i] for i in range(len(terminals))], 'all actual records durably independently matched')
    counts = {k: dict(attempted=0, completed=0) for k in contract()['caps']}
    for r in report['actual_operation_receipts']:
        c = counts[r['resource']]; c['attempted'] += 1; c['completed'] += r['status'] == 'returned'
    require(counts == report['counts'] and all(v['attempted'] <= contract()['caps'][k] for k, v in counts.items()), 'actual meters not fullcount or intent fill')
    closed = [e['receipt'] for e in events if e['kind'] == 'branch_closed']
    require(closed == [r for r in report['rows'] if r['status'] != 'not_started'], 'durable fixed24 failure denominator')
    for i, row in enumerate(report['rows']):
        selected = p['suffix_plans'][i]; original = p['plans'][i//3]
        require(type(row['index']) is type(row['anchor_index']) is int and row['index'] == i and row['anchor_index'] == i//3 and
                row['selected_suffix'] == selected and key(row) == key(original) and row['variant'] == VARIANTS[i%3] and
                row['selection_digest'] == original['selection_digest'], 'own case/root/prefix/winner seal')
        stepops = [r for r in report['actual_operation_receipts'] if r['resource'] == 'public_step' and r.get('branch') == i]
        for phase, commands in (('warmup', base.moving.warmup_commands()), ('plan', base.previous.commands(original['commands'])[:5]),
                                ('suffix', raw5(selected['plan']['suffix']))):
            ops = [r for r in stepops if r['phase'] == phase]; label = 'prefix' if phase == 'plan' else phase
            require([r['step'] for r in ops] == list(range(1, len(ops)+1)) and len(ops) <= len(commands) and
                    all(r['raw_bits'] == commands[j].tobytes().hex() for j, r in enumerate(ops)) and
                    row[label+'_returned'] == sum(r['status'] == 'returned' for r in ops) and
                    row[label+'_attempted'] == len(ops), 'contiguous original/winner raw bits and actual attempt-return')
        responses = row['responses']; require([x['step'] for x in responses] == list(range(1, len(responses)+1)) and len(responses) <= row['suffix_returned'], 'actual responses only')
        if row['suffix_attempted']:
            verified = [e for e in events if e['kind'] == 'suffix_start_verified' and e.get('branch') == i]
            require(len(verified) == 1 and row['start_verified'] is True and row['start_state'] == p['anchors'][i//3]['checkpoint']['state'], 'exact full51 restore BEFORE suffix')
            before = [e for e in events[:events.index(verified[0])] if e['kind'] == 'fresh_observation' and e.get('branch') == i]
            require(len(before) == 51, 'all51 triplet replay observations')
            for actual, expected in zip(before, p['anchors'][i//3]['trace'], strict=True):
                require(actual['pre'] == actual['post'] == expected['pre'] == expected['post'] and
                        all(actual['receipt'][k] == expected['receipt'][k] for k in ('phase', 'step', 'condition', 'overlay_index',
                            'file_sha256', 'pixel_sha256', 'state', 'snapshot')), 'all original51 exact state/body/goal/render/pixels')
        for response in responses:
            state = base.source.vector(response['state']); start = base.source.vector(row['start_state'])
            require(np.array_equal(state-start, response['delta_native7']) and
                    response['native_metrics'] == base.previous.metric(row['goal'], state), 'native7 labels/deltas, no reported fill')
    return deepcopy(report['rows'])


def consume(output, expected):
    """Saved validator only; retains all24 rows, cannot run physics or neural code."""
    base.hash_value(expected); directory = Path(absolute(str(output)))
    files = list(directory.iterdir())
    require(all(f.is_file() and not f.is_symlink() for f in files) and sum(f.stat().st_size for f in files) <= CAP, 'bounded regular physical artifacts')
    rp = directory/'report.json'; require(sha(rp) == expected, 'MAIN external actual physical report SHA')
    r = read(rp, CAP); binding = r['execution_binding']
    require(sha(directory/'execution_binding.json') == r['execution_binding_sha256'] and
            read(directory/'execution_binding.json') == binding, 'physical binding SHA/identity')
    p, _ = verify(binding['prepared'], binding['prepare_sha256'])
    jp = directory/'journal.jsonl'; require(sha(jp) == r['journal_sha256'] and jp.stat().st_size <= CAP, 'journal bytes SHA/size')
    content = jp.read_bytes(); require(len(content) <= CAP and content.endswith(b'\n'), 'complete bounded journal')
    events = [json.loads(line) for line in content.splitlines()]; tail = None
    for index, event in enumerate(events):
        require(event['index'] == index and event['previous_sha256'] == tail and event['event_sha256'] ==
                base.source.digest({k: v for k, v in event.items() if k != 'event_sha256'}), 'actual physical hashchain')
        tail = event['event_sha256']
    require(tail == r['journal_chain_tail'], 'physical terminal tail')
    rows = validate_saved(r, events, p); bindings = deepcopy(p['bindings'])
    from PIL import Image
    for e in events:
        if e['kind'] != 'fresh_observation': continue
        receipt = e['receipt']; name = receipt['png']
        require(Path(name).name == name and name not in ('', '.', '..'), 'flat PNG')
        path = directory/name; require(sha(path) == receipt['file_sha256'], 'actual durable PNG SHA')
        with Image.open(path) as image:
            require(image.size == (224, 224) and image.mode == 'RGB', 'native RGB224')
            pixels = np.asarray(image)
        require(base.moving.image_sha(pixels) == receipt['pixel_sha256'], 'actual pixel SHA')
    merge(bindings, {binding['prepared']: binding['prepare_sha256']})
    merge(bindings, {str(f): sha(f) for f in files})
    require(bindings[str(rp)] == expected, 'final physical report TOCTOU')
    for f, h in bindings.items(): require(sha(f) == h, 'final full saved closure')
    return dict(schema=NAME+'.consume', status=r['status'], rows=rows, fixed_denominator=24,
        counts=r['counts'], bindings=bindings, physical_report_sha256=expected,
        compiler_report_sha256=p['compiler_report_sha256'], consumer_report_sha256=p['consumer_report_sha256'],
        P1_admission=False, scientific_admission=False, independent_confirmation=False, new_neural_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'run', 'consume'))
    for k in ('output', 'prepared', 'prepare-sha', 'report-sha', 'compiler-output', 'compiler-report-sha',
              'producer-source-sha', 'consumer-source-sha', 'consumer-report', 'consumer-report-sha', 'physical-output', 'physical-report-sha'):
        parser.add_argument('--'+k)
    parser.add_argument('--authorize-physics', action='store_true'); a = parser.parse_args()
    if a.command == 'prepare':
        arguments = dict(compiler_output=str(Path(a.compiler_output).resolve()), compiler_report_sha256=a.compiler_report_sha,
            producer_source_sha256=a.producer_source_sha, consumer_source_sha256=a.consumer_source_sha,
            consumer_report=str(Path(a.consumer_report).resolve()), consumer_report_sha256=a.consumer_report_sha,
            physical_output=str(Path(a.physical_output).resolve()), physical_report_sha256=a.physical_report_sha)
        r = prepare(a.output, arguments)
    elif a.command == 'verify':
        p, _ = verify(a.prepared, a.prepare_sha); r = dict(status='verified_no_execution', contract=p['contract'])
    elif a.command == 'consume': r = consume(a.output, a.report_sha)
    else:
        require(a.authorize_physics, 'MAIN CLI explicit physics authorization')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1', render_mode='rgb_array', resolution=224, with_target=False)
        print('24 v2 model-only suffixes after exact original51trace; calibration only, no scientific pass.', flush=True)
        r = run(a.prepared, a.prepare_sha, a.output, factory, authorization=AUTH)
    print(base.source.canonical({k: v for k, v in r.items() if k not in ('rows', 'actual_operation_receipts')}))
    if a.command == 'run' and r['status'] == 'partial_or_failed': raise SystemExit(2)


if __name__ == '__main__': main()
