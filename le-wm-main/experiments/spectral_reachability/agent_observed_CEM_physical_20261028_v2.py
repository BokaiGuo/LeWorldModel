"""Fixed four observed-prefix CEM64 suffixes; MAIN-only physics, saved validation.

No model runner, saved-poststate restore, old endpoint substitution or admission.
External actual producer AND successful own-consume seals are mandatory.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import importlib
import hashlib
import io
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np

from experiments.spectral_reachability import moving_history_observed_suffix_physical_20261027_v1 as observed
from experiments.spectral_reachability import p1_conditional_task_admission_20261027_v1 as core

NAME = 'agent_observed_CEM_physical_20261028_v2'
PRODUCER = 'agent_baselines_resume_20261027_next'
ARM = 'observed_prefix_CEM64'
AUTH = 'MAIN_OBSERVED_CEM64_SUFFIX_PHYSICS_ONLY'
ROOT, base, moving, source = observed.ROOT, observed.base, observed.moving, observed.source
require, sha, read, raw = observed.require, observed.sha, observed.read, observed.raw
HELPER_SHA = '07215161b9ea620efe83ff7ca7ae2838d2f27b40ca16f5cc3aa969626813b2ce'
ORDER = [(s, a) for s in (0, 1) for a in observed.ARMS]
CAP, FLOOR, FINAL, PREP_CAP = 8*1024**2, 200*1024**2, 512*1024, 1024*1024
RECORDED = 'observed_CEM_suffix_physical_recorded_not_admitted'
ARGUMENTS = {'compiler_output', 'compiler_report_sha256', 'producer_source_sha256',
             'consumer_source_sha256', 'consumer_report', 'consumer_report_sha256',
             'consumer_wrapper_source', 'consumer_wrapper_source_sha256',
             'original_physical_report', 'original_physical_report_sha256', 'wall_seconds'}


def hash_value(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None and
            value != '0'*64, 'external actual nonzero SHA256 required')


def contract(wall=600):
    require(type(wall) is int and 0 < wall <= 600, 'declared wall1..600s')
    c = observed.contract(wall)
    c['caps'].update(constructor=4, close=4, goal_setter=12)
    c.update(scope='fixed4 observed-prefix CEM64 suffix calibration; not complete CEM-MPC',
             output_cap_bytes=CAP, disk_floor_bytes=FLOOR, final_reserve_bytes=FINAL,
             old_endpoint_substitution=False, old_control_reexecution=False,
             scientific_admission=False, independent_confirmation=False,
             equal_total_compute=False, equal_FLOPs=False,
             constructor_scope='factory entry/return; internal constructor work not native-hook counted',
             native_scope='after constructor through reset/warm10/prefix5/suffix<=5; hooks restored before close',
             meter_scope='intent, actual callback attempted/completed; failed attempts never refunded',
             wall_scope='run entry before saved verify; predispatch only, blocking calls not preempted')
    return c


def prefix_references(data):
    """Drop all saved suffixes/endpoints/labels from execution inputs."""
    return [dict(root_slot=x['root_slot'], arm=x['arm'],
                 original_selection_digest=x['original_selection_digest'],
                 prefix_commands=deepcopy(x['original_commands'][:5]),
                 warmup=deepcopy(x['warmup']), prefix=deepcopy(x['prefix']),
                 goal=deepcopy(x['goal']), goal_rgb_sha256=x['goal_rgb_sha256'])
            for x in data['paired']]


def validate_plans(consumed, producer_prepare, data, report_sha):
    """Pure RAM ABI seam: actual layout, not a fakecounter or model evaluation."""
    hash_value(report_sha)
    require(consumed['schema'] == PRODUCER+'.consume' and
            consumed['status'] == 'observed_CEM_commands_verified_not_admitted' and
            consumed['errors'] == [] and consumed['new_neural_calls'] == consumed['new_physics_calls'] == 0 and
            consumed['P1_admission'] is consumed['P3_admission'] is consumed['equal_total_compute'] is False,
            'MAIN actual successful own-consume, never raw compiler release alone')
    branches = producer_prepare['branches']
    require(producer_prepare['schema'] == PRODUCER+'.prepare' and
            [(b['source_slot'], b['arm']) for b in branches] == ORDER and
            [(p['source_slot'], p['prefix_arm'], p['arm']) for p in consumed['plans']] ==
            [(s, a, ARM) for s, a in ORDER] and
            [(x['root_slot'], x['arm']) for x in data['paired']] == ORDER,
            'all4 original prefixes in fixed order, never merge by slot/CEM arm')
    rows = []
    for p, b, old in zip(consumed['plans'], branches, data['paired'], strict=True):
        slot = p['source_slot']; require(type(slot) is int, 'literal root slot')
        ref = data['references'][slot]
        body = {k: v for k, v in p.items() if k not in ('selection_sha256', 'whole_run_report_sha256')}
        # Producer uses its own compact core digest, unlike physical source.digest.
        require(p['schema'] == PRODUCER+'.model_plan' and core.digest(body) == p['selection_sha256'] and
                p['whole_run_report_sha256'] == report_sha and p['P1_admission'] is False and
                p['scientific_admission'] is p['independent_confirmation'] is p['equal_total_compute'] is False,
                'sealed own model-only CEM plan and actual whole-run SHA')
        commands = base.commands(p['commands'])
        require(p['commands'] == p['prefix_commands']+p['suffix_commands'] and
                p['prefix_commands'] == b['prefix_commands'] and
                moving.exact(commands[:5], base.commands(old['original_commands'])[:5]),
                'literal original prefix5 and own fullraw10; no clipping/padding')
        require(p['branch_id'] == b['branch_id'] and p['parent_task_id'] == b['parent_task_id'] and
                p['parent_selection_sha256'] == b['planner_selection_sha256'] and
                p['physical_prefix_selection_digest'] == b['physical_selection_digest'] == old['original_selection_digest'] and
                p['blind_sha256'] == b['blind_sha256'] == ref['blind_sha256'] and
                p['observed_context_sha256'] == core.digest(b['observed_context']) and
                p['prefix_npy'] == b['prefix_npy'] and b['checkpoint'] == old['prefix'][5],
                'exact task/root/parent/prefix RGB-state/observed context/NPY identity')
        require(p['horizon'] == 2 and p['reoptimized_horizon'] == 1 and p['raw_steps'] == 10 and
                p['selection'] == 'model_only_stable64_argmin' and
                type(p['selected_index']) is int and 0 <= p['selected_index'] < 64 and
                np.isfinite(p['model_Q']), 'CEM64 H1 winner not physical outcome selection')
        row = dict(root_slot=slot, prefix_arm=p['prefix_arm'], arm=ARM,
                   commands=deepcopy(p['commands']), compiler_plan=deepcopy(p))
        row['selection_digest'] = source.digest(row)
        rows.append(row)
    return rows


def material(arguments):
    require(set(arguments) == ARGUMENTS, 'explicit external authority arguments, no defaults/future hashes')
    a = deepcopy(arguments); c = contract(a['wall_seconds']); bindings = {}
    def bind(path, expected):
        hash_value(expected); path = Path(path)
        require(path.is_absolute() and sha(path) == expected, 'absolute actual input SHA: '+str(path))
        path = str(path.resolve()); require(path not in bindings or bindings[path] == expected, 'consistent closure')
        bindings[path] = expected; return Path(path)
    producer_source = ROOT/'experiments/spectral_reachability'/(PRODUCER+'.py')
    bind(producer_source, a['producer_source_sha256'])
    require(a['consumer_source_sha256'] == a['producer_source_sha256'], 'own consume is in same reviewed producer source')
    wrapper = bind(a['consumer_wrapper_source'], a['consumer_wrapper_source_sha256'])
    require(wrapper.name == 'consume_observed_CEM_MAIN_20261028_v1.py', 'reviewed MAIN saved-consume wrapper; never execute it here')
    bind(Path(observed.__file__).resolve(), HELPER_SHA)
    report_path = bind(Path(a['compiler_output'])/'report.json', a['compiler_report_sha256'])
    accepted_path = bind(a['consumer_report'], a['consumer_report_sha256'])
    original_path = bind(a['original_physical_report'], a['original_physical_report_sha256'])
    require(a['original_physical_report_sha256'] == observed.ORIGINAL_SHA, 'only original fixed4 physical authority')
    producer = importlib.import_module('experiments.spectral_reachability.'+PRODUCER)
    accepted = read(accepted_path)
    require(accepted == producer.consume(str(report_path.parent), a['compiler_report_sha256']),
            'MAIN saved successful own-consume must equal fresh saved-only verifier')
    report = read(report_path); pp = producer.verify(report['prepared'], report['prepare_sha256'])
    data = observed.load_original(original_path, a['original_physical_report_sha256'])
    plans = validate_plans(accepted, pp, data, a['compiler_report_sha256'])
    bindings.update(pp['bindings']); bindings.update(data['bindings'])
    bind(Path(report['prepared']), report['prepare_sha256'])
    for name, h in report['artifact_sha256'].items(): bind(base.flat(report_path.parent, name), h)
    for f in (Path(__file__), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md'),
              Path(base.__file__), Path(base.low.__file__), Path(moving.__file__), Path(source.__file__)):
        bind(f.resolve(), sha(f))
    for path, h in bindings.items(): require(sha(path) == h, 'final complete input closure')
    return dict(schema=NAME+'.prepare', arguments=a, contract=c, plans=plans,
                prefixes=prefix_references(data), sourcepack_sha256=data['original_prepared']['sourcepack_sha256'],
                bindings=bindings, reconstruction='original_seed_placement_warm10_prefix5_exact_exported_trace')


def prepare(output, arguments):
    p = material(arguments); value = raw(p); path = Path(output).absolute()
    require(len(value) <= PREP_CAP and path.parent.is_dir() and not path.exists() and not path.is_symlink() and
            shutil.disk_usage(path.parent).free >= FLOOR+len(value), 'exclusive bounded prepare +200MiB floor')
    with path.open('xb') as f:
        require(f.write(value) == len(value), 'complete prepare'); f.flush(); os.fsync(f.fileno())
    return dict(prepared=str(path), prepare_sha256=sha(path), bytes=len(value))


def verify(path, expected):
    hash_value(expected); require(sha(path) == expected and Path(path).stat().st_size <= PREP_CAP, 'actual physical prepare seal')
    p = read(path); require(p == material(p['arguments']), 'full saved authority/source reconstruction'); return p


class Store(base.Store):
    def check(self, size=4096):
        used = sum(p.stat().st_size for p in self.path.iterdir() if p.is_file())
        require(used+size <= CAP-FINAL and shutil.disk_usage(self.path).free >= FLOOR+size, 'predispatch output cap/floor')
    def write(self, name, value, append=False, final=False):
        require(Path(name).name == name and name not in ('', '.', '..') and not (self.path/name).is_symlink(), 'flat output')
        used = sum(p.stat().st_size for p in self.path.iterdir() if p.is_file())
        require(used+len(value) <= CAP-(0 if final else FINAL) and
                shutil.disk_usage(self.path).free >= FLOOR+len(value), 'output cap/reserve/floor includes failed writes')
        with (self.path/name).open('ab' if append else 'xb') as f:
            require(f.write(value) == len(value), 'complete artifact write'); f.flush(); os.fsync(f.fileno())
        fd = os.open(self.path, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
        return sha(self.path/name)


class Ops:
    def __init__(self, p, store, started=None):
        self.p, self.store = p, store
        self.started = time.monotonic() if started is None else started
        self.where, self.active, self.serial = {}, [], 0
        self.counts = {k: dict(attempted=0, completed=0) for k in p['contract']['caps']}
        self.intents = {k: 0 for k in self.counts}
    def call(self, kind, fn, cleanup=False, **fields):
        require(cleanup is False or kind == 'close', 'only cleanup close bypasses wall/storage precheck')
        require(cleanup or time.monotonic()-self.started < self.p['contract']['wall_seconds'], 'wall before dispatch')
        require(self.intents[kind] < self.p['contract']['caps'][kind], 'cap before dispatch: '+kind)
        if not cleanup: self.store.check(4096)
        n = self.serial; self.serial += 1; self.intents[kind] += 1
        receipt = dict(op_id=n, resource=kind, **self.where, **fields)
        # Cleanup still executes even when durable intent cannot be saved.
        if cleanup:
            try: self.store.emit('operation_intent', **receipt)
            except BaseException: pass_cleanup = True
            else: pass_cleanup = False
        else:
            self.store.emit('operation_intent', **receipt); pass_cleanup = False
        if cleanup:
            try:
                if not pass_cleanup: self.store.emit('operation_attempted', **receipt)
            except BaseException: pass_cleanup = True
        else: self.store.emit('operation_attempted', **receipt)
        self.counts[kind]['attempted'] += 1; self.active.append(kind)
        try:
            value = fn()
        except BaseException as exc:
            try: self.store.emit('operation_failed', **receipt, error_type=type(exc).__name__, message=str(exc))
            except BaseException: pass  # primary exception retained; damaged journal fails saved validation
            raise
        else:
            self.counts[kind]['completed'] += 1
            require(not pass_cleanup, 'cleanup completed but durable operation receipt failed')
            self.store.emit('operation_completed', **receipt)
            return value
        finally: require(self.active.pop() == kind, 'lexical native hook balance')


@contextmanager
def native_hooks(env, ops, row):
    """Unchanged exact replay hooks plus separately metered native goal setter."""
    body = env.unwrapped; old = body._set_goal_state; owned = '_set_goal_state' in body.__dict__
    def goal_setter(value):
        require('reset' in ops.active or 'goal_bind' in ops.active, 'goal setter only in original reset/offline bind')
        return ops.call('goal_setter', lambda: old(value))
    try:
        body._set_goal_state = goal_setter
        with observed.native_hooks(env, ops, row): yield
    finally:
        try:
            if owned: body._set_goal_state = old
            else: delattr(body, '_set_goal_state')
        except BaseException as exc: base.retain(row, exc, 'restore_goal_hook')


def branch(env, ref, source_plan, plan, paired, row, ops, store, index):
    """Original recipe, not reset(savedpoststate); no old suffix or endpoint access."""
    body = env.unwrapped; warm = base.raw_history(ref['raw_history']); cmd = base.commands(plan['commands'])
    require((plan['root_slot'], plan['prefix_arm']) == (paired['root_slot'], paired['arm']) and
            moving.exact(cmd[:5], np.asarray(paired['prefix_commands'], np.float32)), 'original fixed prefix before reset')
    capture = lambda obs, phase, step: observed.paired_capture(env, obs, phase, step, row, ops, store, index, paired)
    store.emit('branch_started', **ops.where, plan=plan, raw_warmup=warm.tolist(), raw_float32_bits=warm.tobytes().hex())
    _, info = ops.call('reset', lambda: env.reset(seed=ref['slot']['reset_seed']), recipe='original_seed')
    original_goal = source.vector(info['goal_state'])
    requested = source.sample_state(original_goal, list(body.block.shapes), dict(root_seed=ref['slot']['root_seed']), 0, 0, source_plan)
    store.emit('placement_request', **ops.where, original_goal=original_goal.tolist(), requested=requested.tolist(), placement_seed=0)
    obs, _ = ops.call('reset', lambda: env.reset(seed=0, options=dict(state=requested.copy(), goal_state=original_goal.copy())), recipe='original_placement')
    frame, snap = capture(obs, 'warmup', 0); history, states, dynamics, sent = [frame], [source.vector(obs['state'])], [snap], []
    for step, command in enumerate(warm, 1):
        require(not body.eval_state(original_goal, source.vector(obs['state']))[0], 'terminal before warmup')
        obs, reward, terminated, truncated, _ = ops.call('public_step', lambda c=command: env.step(c.copy()), phase='warmup', step=step, raw_action=command.tolist())
        row['warmup_returned'] += 1
        store.emit('returned_transition', **ops.where, phase='warmup', step=step, state=source.vector(obs['state']).tolist(), reward=float(reward), terminated=bool(terminated), truncated=bool(truncated))
        frame, snap = capture(obs, 'warmup', step)
        require(moving.exact(np.asarray(body.latest_action), command), 'actual warmup raw bytes')
        sent.append(np.asarray(body.latest_action).copy())
        store.emit('warmup_return', **ops.where, step=step, raw_action=command.tolist(), raw_float32_bits=command.tobytes().hex())
        require(not terminated and not truncated, 'warmup stopped: no padding/retry')
        if step in (5, 10): history.append(frame); states.append(source.vector(obs['state'])); dynamics.append(snap)
    require(moving.exact(np.asarray(sent), warm) and moving.exact(np.asarray(history), ref['history']) and
            moving.exact(np.asarray(states), ref['states']) and dynamics == ref['dynamics'], 'source warm0/5/10 exact')
    goal, rgb = ref['goal'], ref['goal_rgb']; root = source.vector(obs['state'])
    require(source.root_hash(root, goal) == ref['root_sha256'] and goal.tolist() == paired['goal'] and
            moving.image_sha(rgb) == paired['goal_rgb_sha256'], 'source root/goal RGB identity')
    before, pose, counts = moving.snapshot(body), deepcopy(getattr(body, 'goal_pose', None)), deepcopy(ops.counts)
    ops.call('goal_bind', lambda: moving.attach_goal(body, goal, rgb))
    require(moving.snapshot(body) == before and np.array_equal(getattr(body, 'goal_pose', None), pose) and
            all(v == ops.counts[k] for k, v in counts.items() if k not in ('goal_bind', 'goal_setter')),
            'offline bind: no restore/render/physics/goalpose')
    frame, _ = capture(obs, 'plan', 0)
    require(moving.exact(frame, history[-1]) and np.array_equal(body.goal_state, goal) and np.array_equal(body._goal, rgb), 'goal bind exact')
    row.update(initial_exact=True, root_state=root.tolist(), goal=goal.tolist(), initial_metrics=base.metric(goal, root))
    require(not row['initial_metrics']['native_success'], 'not an already-solved root')
    store.emit('goal_attached', **ops.where, goal=goal.tolist(), goal_rgb_sha256=moving.image_sha(rgb), snapshot=before)
    for step, command in enumerate(cmd, 1):
        if step == 6: require(row['prefix_exact'], 'exact prefix admission before suffix')
        def dispatch(c=command):
            row['own_attempted'] += 1
            value = env.step(c.copy()); row['own_returned'] += 1; return value
        obs, reward, terminated, truncated, info = ops.call('public_step', dispatch, phase='plan', step=step, raw_action=command.tolist())
        state = source.vector(obs['state']); metrics = base.metric(goal, state); success, distance = body.eval_state(goal, state)
        row.update(final_state=state.tolist(), final_metrics=metrics, native_success=bool(success), terminated=bool(terminated), truncated=bool(truncated))
        store.emit('returned_transition', **ops.where, phase='plan', step=step, command=command.tolist(), state=state.tolist(), reward=float(reward),
                   terminated=bool(terminated), truncated=bool(truncated), returned_goal_state=np.asarray(info['goal_state']).tolist(), returned_goal_rgb_sha256=moving.image_sha(info['goal']))
        capture(obs, 'plan', step)
        require(moving.exact(np.asarray(body.latest_action), command), 'native raw command substitution')
        require(np.array_equal(info['goal_state'], goal) and np.array_equal(info['goal'], rgb) and
                np.array_equal(body.goal_state, goal) and np.array_equal(body._goal, rgb), 'original goal remains bound')
        require(bool(success) == metrics['native_success'] and float(distance) == metrics['native_distance'] and
                bool(terminated) == bool(success) and float(reward) == -float(distance), 'native7 reward/stop semantics')
        if step <= 5: require(not terminated and not truncated, 'original prefix stopped; NO suffix')
        if step == 5:
            row['prefix_exact'] = row['prefix_goal_exact'] = True
            store.emit('suffix_admitted_after_exact_prefix', **ops.where, prefix_snapshot=moving.snapshot(body), goal=goal.tolist(), goal_rgb_sha256=moving.image_sha(rgb))
        if terminated or truncated:
            row['status'] = 'native_success' if success and terminated and not truncated else 'native_stopped'; break
    else: row['status'] = 'horizon_completed'
    row['full_horizon'] = row['own_returned'] == 10
    row['endpoint_scope'] = 'native_terminal' if row['terminated'] else 'full10' if row['full_horizon'] else 'partial'


def execute(p, data, store, factory, *, authorization, started=None):
    require(authorization == AUTH, 'MAIN-only explicit physics authorization')
    require([(x['root_slot'], x['prefix_arm'], x['arm']) for x in p['plans']] == [(s, a, ARM) for s, a in ORDER] and
            len(data['references']) == 2 and p['prefixes'] == prefix_references(data), 'fixed4 exact reconstruction inputs')
    rows = [dict(root_slot=x['root_slot'], prefix_arm=x['prefix_arm'], arm=ARM, selection_digest=x['selection_digest'],
                 status='not_started', first_error=None, secondary_errors=[], warmup_returned=0, own_attempted=0, own_returned=0,
                 initial_exact=False, prefix_exact=False, prefix_goal_exact=False, full_horizon=False, native_success=None,
                 terminated=None, truncated=None, final_state=None, final_metrics=None, PNG=0, checkpoints=[])
            for x in p['plans']]
    result = dict(schema=NAME+'.report', status='running', rows=rows, errors=[], contract=p['contract'],
                  sourcepack_sha256=p['sourcepack_sha256'], P1_admission=False, P4_admission=False, scientific_admission=False,
                  independent_confirmation=False, equal_total_compute=False, equal_FLOPs=False, new_neural_calls=0, training=False)
    ops = Ops(p, store, started)
    def error(stage, exc): result['errors'].append(dict(stage=stage, type=type(exc).__name__, message=str(exc)))
    try:
        store.emit('run_started', prepare=p)
        for index, (plan, row, paired) in enumerate(zip(p['plans'], rows, p['prefixes'], strict=True)):
            holder = []
            ops.where = dict(branch=index, root_slot=plan['root_slot'], prefix_arm=plan['prefix_arm'], arm=ARM, selection_digest=plan['selection_digest'])
            def construct():
                env = factory(); holder.append(env); return env
            try:
                env = ops.call('constructor', construct); row['status'] = 'running'
                moving.validate_env(env.unwrapped)
                require(env.unwrapped.render_action is False and env.unwrapped.render_size == 224 and
                        env.unwrapped.render_mode == 'rgb_array', 'native target-free RGB224')
                with native_hooks(env, ops, row):
                    branch(env, data['references'][plan['root_slot']], data['plan'], plan, paired, row, ops, store, index)
            except BaseException as exc: base.retain(row, exc, 'setup_or_branch')
            finally:
                if holder:
                    try: ops.call('close', holder[0].close, cleanup=True)
                    except BaseException as exc: base.retain(row, exc, 'close')
                store.emit('branch_closed', receipt=row)
    except BaseException as exc: error('execution', exc)
    try:
        for path, h in p['bindings'].items(): require(sha(path) == h, 'final closure: '+path)
    except BaseException as exc: error('final_bindings', exc)
    clean = not result['errors'] and all(r['status'] in ('horizon_completed', 'native_success', 'native_stopped') and
                r['first_error'] is None and not r['secondary_errors'] and r['prefix_exact'] and r['prefix_goal_exact'] for r in rows)
    if clean:
        try:
            c = ops.counts
            require(all(c[k]['attempted'] == c[k]['completed'] == ops.intents[k] for k in c) and
                    all(c[k]['completed'] == v for k, v in dict(constructor=4, close=4, reset=8, set_state=16, goal_render=8, goal_bind=4, goal_setter=12).items()) and
                    c['public_step']['completed'] == sum(r['warmup_returned']+r['own_returned'] for r in rows) and
                    c['physics_step']['completed'] == 16+10*c['public_step']['completed'] and
                    c['explicit_render']['completed'] == sum(r['PNG'] for r in rows), 'actual independent resource census')
        except BaseException as exc: error('final_census', exc); clean = False
    result.update(status=RECORDED if clean else 'partial_or_failed', counts=deepcopy(ops.counts), intents=deepcopy(ops.intents),
                  full_suffixes=sum(r['full_horizon'] for r in rows), early_stops=sum(r['status'] in ('native_success', 'native_stopped') for r in rows),
                  engineering_failures=sum(r['status'] == 'failed' for r in rows), not_started=sum(r['status'] == 'not_started' for r in rows),
                  native_successes=sum(r['status'] == 'native_success' for r in rows), wall_seconds=time.monotonic()-ops.started)
    if result['wall_seconds'] > p['contract']['wall_seconds']:
        error('final_wall', RuntimeError('wall exceeded; no preemption claim')); result['status'] = 'partial_or_failed'
    try:
        store.emit('run_finished', status=result['status'], counts=ops.counts, intents=ops.intents, errors=result['errors'])
        result['journal_sha256'] = sha(store.path/'journal.jsonl'); result['journal_chain_tail'] = store.previous
    except BaseException as exc: error('final_journal', exc); result['status'] = 'partial_or_failed'
    try: store.put('report.json', result, final=True)
    except BaseException as exc: error('final_report', exc); result['status'] = 'partial_or_failed'
    return result


def run(prepared, expected, output, factory, *, authorization):
    require(authorization == AUTH, 'MAIN physics authorization')
    started = time.monotonic(); p = verify(prepared, expected)
    data = observed.load_original(p['arguments']['original_physical_report'], p['arguments']['original_physical_report_sha256'])
    path = Path(output).absolute()
    require(path.parent.is_dir() and not path.exists() and not path.is_symlink() and
            shutil.disk_usage(path.parent).free >= FLOOR+CAP, 'NEW physical output +200MiB floor')
    store = Store(path); store.put('execution_binding.json', dict(prepared=str(Path(prepared).resolve()), prepare_sha256=expected))
    return execute(p, data, store, factory, authorization=authorization, started=started)


def validate_saved(report, events, p):
    """Pure saved validator. Partial journals must still be intact, never repaired."""
    require(report['schema'] == NAME+'.report' and report['status'] in (RECORDED, 'partial_or_failed') and
            report['contract'] == p['contract'] and report['sourcepack_sha256'] == p['sourcepack_sha256'] and
            report['P1_admission'] is report['P4_admission'] is report['scientific_admission'] is report['independent_confirmation'] is
            report['equal_total_compute'] is report['equal_FLOPs'] is report['training'] is False and report['new_neural_calls'] == 0,
            'physical saved protocol and non-admission')
    require(events[0]['kind'] == 'run_started' and events[0]['prepare'] == p and events[-1]['kind'] == 'run_finished' and
            events[-1]['status'] == report['status'] and events[-1]['errors'] == report['errors'], 'durable begin/end closure')
    previous = None
    for i, e in enumerate(events):
        require(e['index'] == i and e['previous_sha256'] == previous and
                e['event_sha256'] == source.digest({k: v for k, v in e.items() if k != 'event_sha256'}), 'complete saved hashchain')
        previous = e['event_sha256']
    require(previous == report['journal_chain_tail'], 'physical journal tail')
    counts = {k: dict(attempted=0, completed=0) for k in p['contract']['caps']}; intents = {k: 0 for k in counts}
    entries, stack = {}, []
    for e in events:
        if not e['kind'].startswith('operation_'): continue
        kind, n, k = e['kind'], e['op_id'], e['resource']; require(k in counts, 'known resource')
        if kind == 'operation_intent':
            require(n == len(entries), 'unique ordered operation intent'); entries[n] = dict(state='intent', receipt=e)
            intents[k] += 1
        else:
            require(n in entries, 'actual callback requires durable intent'); entry = entries[n]
            for field in ('resource', 'branch', 'root_slot', 'prefix_arm', 'arm', 'selection_digest', 'phase', 'step', 'raw_action'):
                require(e.get(field) == entry['receipt'].get(field), 'operation identity bridge')
            if kind == 'operation_attempted':
                require(entry['state'] == 'intent', 'single actual entry'); entry['state'] = 'attempted'; stack.append(n); counts[k]['attempted'] += 1
            else:
                require(kind in ('operation_completed', 'operation_failed') and entry['state'] == 'attempted' and stack and stack.pop() == n, 'balanced actual return/failure')
                entry['state'] = kind
                if kind == 'operation_completed': counts[k]['completed'] += 1
    require(not stack and all(e['state'] in ('intent', 'operation_completed', 'operation_failed') for e in entries.values()) and
            counts == report['counts'] == events[-1]['counts'] and intents == report['intents'] == events[-1]['intents'] and
            all(intents[k] <= p['contract']['caps'][k] for k in counts), 'actual disjoint resource counts/caps')
    require(len(report['rows']) == 4, 'full4 denominator including failures/not-started')
    for i, (row, plan, old) in enumerate(zip(report['rows'], p['plans'], p['prefixes'], strict=True)):
        identity = {k: plan[k] for k in ('root_slot', 'prefix_arm', 'arm', 'selection_digest')}
        require(all(row[k] == v for k, v in identity.items()) and
                not any(k.startswith('saved_old') or k == 'paired_native_distance_delta' for k in row), 'exact CEM/prefix identity, no old endpoints')
        scoped = [e for e in events if e.get('branch') == i]
        require(all(all(e.get(k) == v for k, v in identity.items()) for e in scoped), 'branch-scoped event identity')
        closed = [e['receipt'] for e in events if e['kind'] == 'branch_closed' and e['receipt']['selection_digest'] == row['selection_digest']]
        if row['status'] == 'not_started': require(not scoped and not closed, 'no hidden work on not-started branch'); continue
        require(closed == [row], 'durable branch close receipt')
        frames = [e['receipt'] for e in scoped if e['kind'] == 'fresh_observation']
        require(row['PNG'] == len(frames) and row['checkpoints'] == [f for f in frames if f['phase'] == 'plan' and f['step'] in (0, 5, 10)], 'fresh RGB/checkpoint join')
        for f in frames:
            require(f['state'] == f['observation_state'] == f['snapshot']['state'] and len(f['state']) == 7, 'actual7 state/body bridge')
        admissions = [e for e in scoped if e['kind'] == 'suffix_admitted_after_exact_prefix']
        suffixes = [e for e in scoped if e['kind'] == 'operation_attempted' and e['resource'] == 'public_step' and e.get('phase') == 'plan' and e['step'] > 5]
        if suffixes or row['prefix_exact']:
            require(len(admissions) == 1 and row['prefix_exact'] and row['prefix_goal_exact'] and all(e['index'] > admissions[0]['index'] for e in suffixes), 'admit exact prefix BEFORE any new suffix')
            checks = [e for e in scoped if e['kind'] == 'paired_prefix_exact']
            require([(e['phase'], e['step']) for e in checks] == [('warmup', n) for n in range(11)]+[('plan', n) for n in range(6)] and
                    all(e['accepted'] is True for e in checks), 'all17 paired frame checks')
            for f in frames:
                if f['phase'] == 'warmup' or f['step'] <= 5:
                    r = old['warmup' if f['phase'] == 'warmup' else 'prefix'][f['step']]
                    require(all(f[k] == r[k] for k in ('file_sha256', 'pixel_sha256', 'state', 'snapshot', 'observation_state')), 'original exact frame/state/body bits')
        transitions = [e for e in scoped if e['kind'] == 'returned_transition' and e['phase'] == 'plan']
        attempted = [e for e in scoped if e['kind'] == 'operation_attempted' and e['resource'] == 'public_step' and e.get('phase') == 'plan']
        require(row['own_attempted'] == len(attempted) and row['own_returned'] >= len(transitions) and
                [e['step'] for e in attempted] == list(range(1, len(attempted)+1)) and
                [e['raw_action'] for e in attempted] == plan['commands'][:len(attempted)], 'literal actual own commands/no retry or padding')
        for t in transitions:
            metrics = base.metric(old['goal'], t['state'])
            require(t['command'] == plan['commands'][t['step']-1] and t['returned_goal_state'] == old['goal'] and
                    t['returned_goal_rgb_sha256'] == old['goal_rgb_sha256'] and t['reward'] == -metrics['native_distance'] and
                    t['terminated'] == metrics['native_success'], 'own native7 reward/goal/termination semantics')
        if row['final_metrics'] is not None:
            require(transitions and row['final_state'] == transitions[-1]['state'] and row['final_metrics'] == base.metric(old['goal'], row['final_state']), 'new endpoint exclusively from own returned transition')
        if row['status'] in ('horizon_completed', 'native_success', 'native_stopped'):
            require(row['first_error'] is None and not row['secondary_errors'] and row['prefix_exact'] and row['initial_exact'] and
                    row['warmup_returned'] == 10 and row['own_returned'] == len(transitions) and
                    row['full_horizon'] == (len(transitions) == 10) and 6 <= len(transitions) <= 10 and
                    [(f['phase'], f['step']) for f in frames] == [('warmup', n) for n in range(11)]+[('plan', n) for n in range(len(transitions)+1)], 'complete or native early-stop chronology')
            require(all(not t['terminated'] and not t['truncated'] for t in transitions[:-1]), 'never dispatch after native stop')
            require(row['native_success'] == transitions[-1]['terminated'] and row['terminated'] == transitions[-1]['terminated'] and
                    row['truncated'] == transitions[-1]['truncated'] and
                    (row['status'] == 'horizon_completed') == (len(transitions) == 10 and not row['terminated'] and not row['truncated']) and
                    (row['status'] == 'native_success') == (row['terminated'] and not row['truncated']), 'native status and endpoint are not selected/refilled')
            guards = [e for e in scoped if e['kind'] == 'guard_receipt']
            require([(g['phase'], g['step']) for g in guards] == [(f['phase'], f['step']) for f in frames] and
                    all(g['accepted'] is g['render_preserved_dynamics'] is True and g['canonical_agent_pixels'] > 0 for g in guards), 'all actual RGB/canvas/pure-render guards')
            warms = [e for e in scoped if e['kind'] == 'warmup_return']
            warmraw = moving.warmup_commands()
            require([e['step'] for e in warms] == list(range(1, 11)) and
                    [e['raw_float32_bits'] for e in warms] == [c.tobytes().hex() for c in warmraw] and
                    moving.exact(np.asarray([e['raw_action'] for e in warms], np.float32), warmraw), 'all10 actual warmup raw history')
            plan_frames = [f for f in frames if f['phase'] == 'plan']
            require(row['root_state'] == plan_frames[0]['state'] and row['goal'] == old['goal'] and
                    row['initial_metrics'] == base.metric(old['goal'], row['root_state']) and
                    all(t['state'] == plan_frames[n+1]['state'] for n, t in enumerate(transitions)), 'actual root/goal/transition/frame7 joins')
    summaries = dict(full_suffixes=sum(r['full_horizon'] for r in report['rows']),
                     early_stops=sum(r['status'] in ('native_success', 'native_stopped') for r in report['rows']),
                     engineering_failures=sum(r['status'] == 'failed' for r in report['rows']),
                     not_started=sum(r['status'] == 'not_started' for r in report['rows']),
                     native_successes=sum(r['status'] == 'native_success' for r in report['rows']))
    require(all(report[k] == v for k, v in summaries.items()), 'all4 outcome/failure denominators')
    if report['status'] == RECORDED:
        require(report['errors'] == [] and all(r['status'] in ('horizon_completed', 'native_success', 'native_stopped') for r in report['rows']) and
                all(counts[k]['attempted'] == counts[k]['completed'] == intents[k] for k in counts), 'complete engineering closure only')
        require(all(counts[k]['completed'] == v for k, v in dict(constructor=4, close=4, reset=8, set_state=16, goal_render=8, goal_bind=4, goal_setter=12).items()) and
                counts['public_step']['completed'] == sum(r['warmup_returned']+r['own_returned'] for r in report['rows']) and
                counts['physics_step']['completed'] == 16+10*counts['public_step']['completed'] and
                counts['explicit_render']['completed'] == sum(r['PNG'] for r in report['rows']) and
                0 <= report['wall_seconds'] <= p['contract']['wall_seconds'], 'full engineering resource/wall reconciliation')
    return dict(schema=NAME+'.consume', status='saved_physical_verified_not_admitted' if report['status'] == RECORDED else 'saved_partial_verified_not_admitted',
                rows=deepcopy(report['rows']), counts=counts, scientific_admission=False, P1_admission=False, P4_admission=False,
                independent_confirmation=False, equal_total_compute=False, equal_FLOPs=False, new_neural_calls=0, new_physics_calls=0)


def validate_png(payload, receipt):
    """Memory seam for actual file/pixel validation; no renderer invocation."""
    from PIL import Image
    require(len(payload) <= 1024**2 and hashlib.sha256(payload).hexdigest() == receipt['file_sha256'], 'fresh own PNG byte seal')
    with Image.open(io.BytesIO(payload)) as im:
        require(im.mode == 'RGB' and im.size == (224, 224), 'native RGB224')
        require(moving.image_sha(np.asarray(im)) == receipt['pixel_sha256'], 'fresh own pixel SHA')


def consume(output, expected):
    hash_value(expected); output = Path(output).absolute(); report_path = output/'report.json'
    require(sha(report_path) == expected and report_path.stat().st_size <= FINAL, 'actual bounded physical report SHA')
    report = read(report_path); binding = read(output/'execution_binding.json'); p = verify(binding['prepared'], binding['prepare_sha256'])
    require(sha(output/'journal.jsonl') == report['journal_sha256'], 'actual physical journal SHA')
    events = base.journal(output/'journal.jsonl', report['journal_chain_tail']); result = validate_saved(report, events, p)
    names = {'report.json', 'journal.jsonl', 'execution_binding.json'}
    for e in events:
        if e['kind'] == 'invalid_render':
            r = e['receipt']
            if r['raw_render'] is not None:
                path = base.flat(output, r['raw_render']); names.add(r['raw_render'])
                require(sha(path) == r['file_sha256'] and path.stat().st_size <= CAP, 'sealed invalid render retained only as failure evidence')
                with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as z:
                    require(z.files == ['raw_render'] and list(z['raw_render'].shape) == r['shape'] and
                            str(z['raw_render'].dtype) == r['dtype'], 'invalid render array identity')
            continue
        if e['kind'] != 'fresh_observation': continue
        r = e['receipt']; path = base.flat(output, r['png']); names.add(r['png'])
        require(sha(path) == r['file_sha256'] and path.stat().st_size <= 1024**2, 'fresh own PNG file SHA')
        validate_png(path.read_bytes(), r)
    require(all(f.is_file() and not f.is_symlink() for f in output.iterdir()) and
            sum(f.stat().st_size for f in output.iterdir()) <= CAP, 'bounded actual physical output census')
    require({f.name for f in output.iterdir()} == names, 'no unaccounted saved artifacts, including partial writes')
    require(sha(report_path) == expected, 'final report seal')
    return dict(result, whole_run_report_sha256=expected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'run', 'consume'))
    for k in ('compiler-output', 'compiler-report-sha256', 'producer-source-sha256', 'consumer-source-sha256', 'consumer-report',
              'consumer-wrapper-source', 'consumer-wrapper-source-sha256',
              'consumer-report-sha256', 'original-physical-report', 'original-physical-report-sha256', 'prepared', 'prepare-sha', 'output', 'report-sha'):
        parser.add_argument('--'+k)
    parser.add_argument('--wall-seconds', type=int, default=600); parser.add_argument('--authorize-physics', action='store_true')
    a = parser.parse_args(argv)
    if a.command == 'prepare': result = prepare(a.output, {k: getattr(a, k) for k in ARGUMENTS})
    elif a.command == 'verify': result = dict(status='verified_no_execution', schema=verify(a.prepared, a.prepare_sha)['schema'])
    elif a.command == 'consume': result = consume(a.output, a.report_sha)
    else:
        require(a.authorize_physics, 'MAIN CLI authorization')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1', render_mode='rgb_array', resolution=224, with_target=False, render_action=False)
        result = run(a.prepared, a.prepare_sha, a.output, factory, authorization=AUTH)
    print(source.canonical(result))
    return 2 if result.get('status') == 'partial_or_failed' else 0


if __name__ == '__main__': raise SystemExit(main())
