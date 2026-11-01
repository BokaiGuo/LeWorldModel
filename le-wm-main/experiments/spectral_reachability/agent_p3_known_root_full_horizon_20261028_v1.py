"""MAIN-only known-root P3 full-commit public20 CUDA/native calibration.

Same-process P4 native/visual machinery; model code receives causal whitelisted
inputs only. No neural/physics imports during prepare/verify/saved consumption.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.spectral_reachability import agent_p3_child_signal_cuda_20261028_v2 as paid
from experiments.spectral_reachability import agent_p3_native_program_20261028_v1 as original
from experiments.spectral_reachability import agent_p3_child_native_physical_20261028_v1 as physical
from experiments.spectral_reachability import agent_p4_rolling_saved_consume_20261028_v1 as rolling
from experiments.spectral_reachability import agent_p3_prospective_equal_horizon_20261028_v1 as previous

core, p4, old = paid.core, physical.p4, physical.old
ROOT = core.ROOT
NAME = 'agent_p3_known_root_full_horizon_20261028_v1'
AUTH = 'MAIN_KNOWN_P3_FULL_HORIZON_CUDA_AND_PHYSICS'
MODES = ('fixed', 'adaptive')
CAP, FINAL, FLOOR = 512*1024**2, 32*1024**2, 200*1024**2
WALL = 900
require, sha, digest, canonical = core.require, paid.sha, core.digest, core.canonical
read = paid.base.read
FLAGS = dict(P1_admission=False, scientific_admission=False, C3_demonstrated=False,
             freshness=False, independent_confirmation=False, equal_total_compute=False)


def protocol():
    return dict(schema=NAME+'.protocol', roots=2, episodes=4, modes=list(MODES),
        prefix_arm=old.ARMS[0], warmup=10, common_prefix=5, public_target=20,
        feedback=[5, 10, 15, 20], paid_signal=[5, 10, 15],
        commitment='execute ALL 5H raw actions; observe and pay signal every5; no replacement inside H3',
        scheduling='original core.choose at depth0 only when full previous program returned',
        natural_candidates=[4, 8], quota_override=False, recursive_tree=False,
        residual='whole-node refusal if 5H>20-boundary; no forced H1, clipping, padding or reselection',
        unsupported='original reconfiguration/composition has no native generator: whole-node refusal',
        comparison='same intended public20; pairwise20 metrics ONLY if both arms complete20',
        partial='retain four denominator and actual step/flags/metrics; never compare different terminal times as public20',
        compiler=original.rule(), selected_replay='independent full-H whole AND segmented, no raw5 release',
        FLOPs=None, total_compute=None, device_time_ns=None, inherited_acquisition_cost=None,
        wall_seconds_cap=WALL, output_bytes_cap=CAP, final_reserve_bytes=FINAL, disk_floor_bytes=FLOOR,
        cuda_allocated_bytes=3*1024**3, cuda_reserved_bytes=4*1024**3, CPU_neural=False,
        TF32=False, SDP='MATH', **FLAGS)


def program_caps(h, n):
    require(type(h) is int and h in (1, 3) and type(n) is int and n in (4, 8), 'natural H/N')
    out = previous.predict_axes(h*(2*n+2))
    out.update(array_SVD_calls=1, solve_calls=1, program_compile_calls=1,
               replay_calls=2*n+2, metric_calls=n)
    return out


def arm_caps(mode):
    require(mode in MODES, 'fixed/adaptive')
    n = 4 if mode == 'fixed' else 8
    out = previous.predict_axes(24+3*(2*n+2))
    out.update(jacobian_API=6, VJP_directions=1152, grad_API=1152, raw_backward_callbacks=1152,
        array_SVD_calls=9, solve_calls=3, signal_compile_calls=6, program_compile_calls=3,
        replay_calls=3*(2*n+2), metric_calls=3*n, schedule_decisions=3)
    return out


def native_caps():
    # Reset may include constructor virtual reset: cap3, not assumed actual3.
    return dict(constructor=4, close=4, reset=12, set_state=24, goal_render=12,
        goal_setter=16, goal_bind=4, public_step=120, physics_step=1224,
        explicit_render=128, native_metric=80, encoder=16, projector=16)


class Store(old.Store):
    """New bounded writer, inherited PNG/array/namespace/chain ABI; no globals."""
    def __init__(self, path):
        self.path = Path(path).absolute()
        require(not self.path.exists() and not self.path.is_symlink(), 'new exclusive run directory')
        require(shutil.disk_usage(self.path.parent).free >= FLOOR+CAP, '512MiB reserve plus200MiB floor')
        self.path.mkdir()
        paid.base.fsync_dir(self.path.parent)
        self.previous, self.index = None, 0

    def check(self, size=4096, final=False):
        used = sum(p.stat().st_size for p in self.path.rglob('*') if p.is_file())
        require(used+size <= CAP-(0 if final else FINAL) and
                shutil.disk_usage(self.path).free >= FLOOR+size, 'whole output cap/floor')

    def write(self, name, value, append=False, final=False):
        parts = Path(name).parts
        require(1 <= len(parts) <= 2 and not Path(name).is_absolute() and
            all(x not in ('', '.', '..') for x in parts) and
            not (self.path/name).is_symlink() and not (self.path/name).parent.is_symlink(), 'own artifact path')
        require(not append or name == 'journal.jsonl', 'journal-only append')
        self.check(len(value), final)
        with (self.path/name).open('ab' if append else 'xb') as f:
            require(f.write(value) == len(value), 'complete durable write')
            f.flush()
            os.fsync(f.fileno())
        paid.base.fsync_dir((self.path/name).parent)
        return sha(self.path/name) if not append else None

    png_bytes = p4.Store.png_bytes


class Meter(old.Meter):
    def __init__(self, store, check):
        super().__init__(store, check)
        self.caps = native_caps()
        self.counts = {k: dict(attempted=0, completed=0) for k in self.caps}

    def check(self):
        self.job_check()
        require(time.monotonic()-self.started < WALL, 'native job deadline')
        self.store.check()


class Arm:
    def __init__(self, mode):
        self.caps = arm_caps(mode)
        self.attempted = dict.fromkeys(core.AXES, 0)
        self.completed = deepcopy(self.attempted)

    def reserve(self, change):
        require(all(self.attempted[k]+change[k] <= self.caps[k] for k in core.AXES), 'whole-node arm cap before dispatch')

    def receipt(self):
        return deepcopy(dict(caps=self.caps, attempted=self.attempted, completed=self.completed))


class Trace:
    def __init__(self, store, group, arm):
        self.store, self.group, self.arm, self.events = store, group, arm, []

    def event(self, event):
        if event['kind'] == 'resource_attempt':
            d = event['entry']['delta']
            self.arm.reserve(d)
            for k in core.AXES:
                self.arm.attempted[k] += d[k]
        elif event['kind'] == 'resource_return':
            for k in core.AXES:
                self.arm.completed[k] += event['entry']['delta'][k]
        self.store.emit('model_trace', group=self.group, payload=event)
        self.events.append(deepcopy(event))


def child_packet(eid, mode, step, context, raw10, observation):
    core.context(context)
    require(step in (5, 10, 15) and mode in MODES, 'actual causal boundary')
    require(observation['boundary'] == step and observation['observed_index'] < observation['encoded_index'], 'actual RGB then encoding')
    return dict(child_id=eid+f'_b{step:02}', context=deepcopy(context), context_sha256=digest(context),
        mode=mode, boundary=step, raw_history10=np.asarray(raw10, np.float32).tolist(),
        observation=deepcopy(observation))


def check_child(task, child):
    core.keys(child, ('child_id', 'context', 'context_sha256', 'mode', 'boundary', 'raw_history10', 'observation'))
    require(child == child_packet(child['child_id'].rsplit('_b', 1)[0], child['mode'], child['boundary'],
            child['context'], child['raw_history10'], child['observation']), 'exact live whitelist')
    require(child['context']['action_history'] == p4.normalized(task, child['raw_history10']), 'causal actual raw10 normalization')
    require(child['observation']['observed_latent_sha256'] == old.packet_digest(child['context']['latent_history'][-1]), 'own observed latent, not predicted substitute')
    return deepcopy(child)


class LiveSignal(paid.NativeChildSignal):
    """Replace ONLY old saved roster constructor guard with live RGB/context gate.

    The inherited paid.signal/checked_request math and full tensor hooks are
    unchanged. producer_sha256 names that actual method owner; adapter separately
    source-bound here. No fake old episode/child ancestry is constructed.
    """
    def __init__(self, model, device, task, child, sink):
        self.child = check_child(task, child)
        paid.base.NativeSignal.__init__(self, model, device, task, sink)
        self.source_sha256 = sha(paid.__file__)


def signal_round(task, child, model, device, store, arm, *, decide):
    child = check_child(task, child)
    schedules = int(decide)
    caps = paid.contract.signal_caps(schedules)
    arm.reserve(caps)
    group = child['child_id']+':'+child['mode']
    sink = Trace(store, group, arm)
    ledger = core.Ledger(caps, sink=sink.event)
    ledger.scope = dict(child_id=child['child_id'], group=child['mode'], phase='child_signal', parent_context_sha256=child['context_sha256'])
    acquisition = dict(group=group, status='attempted', signal=None, resource_receipt=None,
        error=None, backend_kind='HOST_CUDA')
    decision = None
    try:
        backend = LiveSignal(model, device, task, child, sink)
        node = dict(context=child['context'], depth=0)
        signals, receipt = core.local_signal(task, node, backend, ledger, core.policy(task, node))
        acquisition['signal'] = receipt
        if decide:
            ledger.scope = dict(child_id=child['child_id'], group=child['mode'], phase='schedule', mode=child['mode'], parent_context_sha256=child['context_sha256'])
            decision = ledger.call('schedule', lambda: core.choose(task, node, signals, child['mode']))
        require(ledger.attempted == ledger.completed == caps, 'exact live paid signal and schedule')
        acquisition['status'] = 'returned'
    except BaseException as exc:
        acquisition.update(status='failed', error=repr(exc))
        raise
    finally:
        acquisition['resource_receipt'] = ledger.receipt()
        store.put(child['child_id']+'_signal.json', dict(child=child, acquisition=acquisition, decision=decision))
    # Same original full-array strict validator, not a new counter-only gate.
    paid.validate_trace(task, child, acquisition, sink.events, schedules)
    return dict(child=child, acquisition=acquisition, decision=decision)


def live_task(task, context):
    out = deepcopy(core.validate_task(task))
    out['root_context'] = deepcopy(context)
    out['binding']['context_sha256'] = digest(context)
    return core.validate_task(out)


def bank(task, response, n, ledger=None):
    h = response['horizon']
    core.finite(response['jacobian'], (192, 10*h), True)
    core.finite(response['nominal'], (192,), True)
    q = np.asarray(task['projection'], float)
    j = q@np.asarray(response['jacobian'], float)
    residual = q@(np.asarray(task['goal'])-np.asarray(response['nominal']))
    def solve():
        return original.generator.bounded_ridge(j, residual,
            damping=original.rule()['damping'], cap=original.rule()['joint_raw_l2_cap'])
    result = solve() if ledger is None else ledger.call('solve', lambda: ledger.call('array_svd', solve))
    direction = np.asarray(result['params'], np.float64)
    divisor = max(1., float(np.max(np.abs(direction))))
    fractions = [i/(n-1) for i in range(n)]
    return dict(solve=result, divisor=divisor, fractions=fractions,
        actions=[(f*direction/divisor).astype(np.float32).reshape(5*h, 2).tolist() for f in fractions])


def compile_full(task, child, signal, model, device, store, arm):
    d = signal['decision']
    require(d['extension_type'] == d['program_type'] == 'progress' and d['continue_expanding'] is True, 'native progress only')
    h, n = d['horizon'], d['candidates']
    caps = program_caps(h, n)
    arm.reserve(caps)
    sink = Trace(store, child['child_id']+':program', arm)
    ledger = core.Ledger(caps, sink=sink.event)
    t = live_task(task, child['context'])
    backend = original.NativeProgram(model, device, t, sink)
    response = signal['acquisition']['signal']['responses'][(h-1)//2]
    require(response['parent_context_sha256'] == child['context_sha256'] and response['binding_sha256'] == digest(task['binding']), 'same paid child J')
    result = dict(child_sha256=digest(child), signal_sha256=digest(signal), decision=deepcopy(d),
        bank=None, rows=[], selected=None, resource_receipt=None, error=None)
    def scope(k):
        return dict(slot=0, mode=child['mode'], index=k, horizon=h)
    def replay_row(k, actions):
        ledger.scope = scope(k)
        whole = ledger.call('replay', lambda: backend.replay(child['context'], actions, False, ledger))
        segmented = ledger.call('replay', lambda: backend.replay(child['context'], actions, True, ledger))
        require(whole == segmented, 'all full-H replay boundaries exact')
        core.checked_path(task, child['context'], actions, whole)
        return dict(slot=0, mode=child['mode'], index=k, horizon=h, actions=deepcopy(actions), whole=whole, segmented=segmented)
    try:
        ledger.scope = scope(None)
        result['bank'] = ledger.call('program_compile', lambda: bank(task, response, n, ledger))
        for k, actions in enumerate(result['bank']['actions']):
            row = replay_row(k, actions)
            row['cost'] = ledger.call('metric', lambda: core.model_cost(task, row['whole'][-1]))
            result['rows'].append(row)
        winner = min(result['rows'], key=lambda r: (r['cost'], r['index']))
        selected = replay_row(n, winner['actions'])
        require(selected['whole'] == winner['whole'], 'independent selected FULL H, not first5')
        result['selected'] = dict(index=winner['index'], actions=winner['actions'], replay=selected)
        require(ledger.attempted == ledger.completed == caps, 'strict full-H compiler budget')
    except BaseException as exc:
        result['error'] = repr(exc)
        raise
    finally:
        result['resource_receipt'] = ledger.receipt()
        store.put(child['child_id']+'_program.json', result)
    validate_program(task, child, signal, result, sink.events)
    return result


def validate_program(task, child, signal, result, events):
    d = signal['decision']
    h, n = d['horizon'], d['candidates']
    response = signal['acquisition']['signal']['responses'][(h-1)//2]
    require(result['child_sha256'] == digest(child) and result['signal_sha256'] == digest(signal) and result['decision'] == d and result['error'] is None, 'same current signal/compiler identity')
    require(canonical(result['bank']) == canonical(bank(task, response, n)), 'independent original ridge/fraction bank recomputation')
    receipt = result['resource_receipt']
    core.audit_ledger(receipt)
    require(receipt['attempted'] == receipt['completed'] == receipt['caps'] == program_caps(h, n), 'exact full-H ledger')
    rows = result['rows']
    require(len(rows) == n and [r['index'] for r in rows] == list(range(n)), 'all original candidate indices')
    for k, row in enumerate(rows):
        require(row['actions'] == result['bank']['actions'][k] and row['whole'] == row['segmented'], 'bank/candidate exact raw and replay')
        core.checked_path(task, child['context'], row['actions'], row['whole'])
        require(row['cost'] == core.model_cost(task, row['whole'][-1]), 'original Q recomputed, no native metric selection')
    w = min(rows, key=lambda r: (r['cost'], r['index']))
    selected = result['selected']
    require(selected['index'] == w['index'] and selected['actions'] == w['actions'] and
        len(selected['actions']) == 5*h and selected['replay']['index'] == n and
        selected['replay']['actions'] == w['actions'] and selected['replay']['whole'] ==
        selected['replay']['segmented'] == w['whole'], 'stable selection and independent full-H identity')
    allowed = {'resource_attempt', 'resource_return', 'io_input', 'io_return', 'program_packet', 'program_boundary'}
    require(all(e['kind'] in allowed for e in events), 'closed original full-buffer trace')
    original.validate_arrays(dict(tasks=[live_task(task, child['context'])]), rows+[selected['replay']], events, receipt)


def roster():
    return [dict(episode_id=f'k{s}_{m}', source_slot=s, prefix_arm=old.ARMS[0], mode=m,
        status='not_started', first_error=None, secondary_errors=[], warmup_returned=0, control_returned=0,
        executed_raw=[], encodings=[], checkpoints=[], PNG=0, initial_exact=False, full_horizon=False,
        native_success=None, terminated=None, truncated=None, rounds=[], resources=None,
        encoder_counts=None) for s in (0, 1) for m in MODES]


def packet(env, commands, row, ref, meter, store, *, seal=None):
    commands = old.raw5(commands)
    start = row['control_returned']
    if start >= 5:
        require(seal is not None and seal['seal_sha256'] == digest({k:v for k,v in seal.items() if k != 'seal_sha256'}), 'durable full-program seal before suffix')
        offset = start-seal['boundary']
        require(0 <= offset and offset+5 <= len(seal['actions']) and
            np.array_equal(commands, np.asarray(seal['actions'], np.float32)[offset:offset+5]), 'own committed full-H slice, never substitute')
    meter.region = 'prefix' if start == 0 else 'suffix'
    for action in commands:
        step = row['control_returned']+1
        store.emit('authorized_dispatch', episode_id=row['episode_id'], step=step,
            seal_sha256=None if seal is None else seal['seal_sha256'], command=action.tolist())
        obs, reward, terminated, truncated, info = env.step(action.copy())
        row['control_returned'] += 1
        actual = np.asarray(env.unwrapped.latest_action).copy()
        require(old.kernel.moving.exact(actual, action), 'literal actual raw action bits')
        row['executed_raw'].append(actual.tolist())
        require(np.array_equal(info['goal_state'], ref['goal']) and np.array_equal(info['goal'], ref['goal_rgb']), 'original goal unchanged')
        meter.call('native_metric', lambda: old.native_transition(env.unwrapped, ref['goal'], obs, reward, terminated, truncated, row))
        _, frame = old.capture(env, obs, row, store, 'plan', step)
        store.emit('control_return', episode_id=row['episode_id'], step=step, command=actual.tolist(),
            reward=float(reward), terminated=bool(terminated), truncated=bool(truncated), state=np.asarray(obs['state']).tolist())
        if terminated or truncated:
            row['status'] = 'native_stopped_control'
            return None
    return frame


def ancestor_gate(store, row, expected):
    actual = [e['receipt'] for e in physical.journal(store.path/'journal.jsonl', CAP)
        if e['kind'] == 'fresh_observation' and e['episode_id'] == row['episode_id'] and
        (e['receipt']['phase'] == 'warmup' or e['receipt']['step'] <= 5)]
    require(len(actual) == len(expected) == 17, 'all warm0..10 and plan0..5 ancestry')
    for a, b in zip(actual, expected, strict=True):
        require((a['phase'], a['step']) == (b['phase'], b['step']), 'original phase/order')
        physical.same_frame(a, b)


def episode(env, ref, source_plan, layer, task, ancestors, row, runtime, meter, store):
    if not p4.setup(env, ref, source_plan, row, meter, store):
        return
    frame = packet(env, layer['commands'][:5], row, ref, meter, store)
    if frame is None:
        return
    ancestor_gate(store, row, ancestors)
    store.emit('ancestor_prefix_verified', episode_id=row['episode_id'], frame_count=17)
    context = deepcopy(task['gn_task']['root_context'])
    arm = Arm(row['mode'])
    pending, seal = [], None
    before_encoder = {k: deepcopy(meter.counts[k]) for k in ('encoder', 'projector')}
    try:
        for boundary in (5, 10, 15, 20):
            require(row['control_returned'] == boundary, 'no mixed horizon feedback')
            raw10 = np.concatenate((ref['raw_history'], np.asarray(row['executed_raw'], np.float32)))[-10:]
            frozen_state = old.state_seal(env.unwrapped)
            z, observation = p4.encode_boundary(old.Handler(None), runtime, task, layer, row,
                frame, env.unwrapped, meter, store, boundary)
            context = p4.roll(context, z, raw10, task['gn_task'])
            require(old.state_seal(env.unwrapped) == frozen_state, 'model side cannot change physical state')
            if boundary == 20:
                require(not pending, 'all selected H3 actions returned before20')
                row.update(full_horizon=True, status='completed20', actual_Q20=core.model_cost(task['gn_task'], context))
                break
            child = child_packet(row['episode_id'], row['mode'], boundary, context, raw10, observation)
            meter.region = 'compiling'
            feedback = signal_round(task['gn_task'], child, runtime['model'], runtime['device'], store, arm, decide=not pending)
            turn = dict(boundary=boundary, signal=feedback, program=None, seal=None, status='paid_feedback')
            row['rounds'].append(turn)
            if not pending:
                d = feedback['decision']
                if 5*d['horizon'] > 20-boundary:
                    turn['status'] = row['status'] = 'refused_full_H_exceeds_remaining'
                    break
                if d['extension_type'] != 'progress' or d['program_type'] != 'progress' or not d['continue_expanding']:
                    turn['status'] = row['status'] = 'refused_undefined_reconfiguration'
                    break
                program = compile_full(task['gn_task'], child, feedback, runtime['model'], runtime['device'], store, arm)
                turn['program'] = program
                pending = deepcopy(program['selected']['actions'])
                seal = dict(schema=NAME+'.dispatch', episode_id=row['episode_id'], boundary=boundary,
                    child_sha256=digest(child), program_sha256=digest(program), actions=deepcopy(pending))
                seal['seal_sha256'] = digest(seal)
                store.put(child['child_id']+'_dispatch.json', seal)
                store.emit('full_program_sealed', episode_id=row['episode_id'], seal=seal)
                turn['seal'] = seal
            else:
                turn['status'] = 'paid_feedback_inside_commit'
            require(old.state_seal(env.unwrapped) == frozen_state, 'signal/compiler/strict cannot alter native state/goal')
            chunk, pending = pending[:5], pending[5:]
            frame = packet(env, chunk, row, ref, meter, store, seal=seal)
            if frame is None:
                break
    finally:
        row['resources'] = arm.receipt()
        row['encoder_counts'] = {k:{s:meter.counts[k][s]-before_encoder[k][s] for s in ('attempted', 'completed')} for k in before_encoder}


def load_data(p):
    manifest = read(p['original_manifest'])
    data = old.kernel.load_source(manifest['sourcepack'], old.SOURCE_SHA, manifest['source_prepared'], manifest['source_prepare_sha256'])
    from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as encoder
    data['tasks'] = {}
    for layer in p['layers']:
        interface = encoder.load_task(layer['task_export'], layer['task_sha256'])
        t = p['tasks'][layer['source_slot']]
        require(interface['gn_task'] == {k:t[k] for k in interface['gn_task']}, 'original model task numerical ABI')
        require(t['binding']['root_sha256'] == data['references'][layer['source_slot']]['root_sha256'] and
            t['binding']['task_sha256'] == layer['task_sha256'], 'native root/task identity')
        data['tasks'][layer['source_slot']] = dict(gn_task=t, model_bindings=encoder.MODELS, modelbinding=p['modelbinding'])
    return data


def build():
    # Existing actual source, not first5 results relabelled as a new execution.
    require(sha(previous.TASK_AUTHORITY) == previous.TASK_AUTHORITY_SHA, 'original tasks authority')
    authority = read(previous.TASK_AUTHORITY)
    tasks = [core.validate_task(t) for t in authority['wiring']['tasks']]
    original_p = old.verify(p4.OLD_PREPARE, p4.OLD_SHA)
    require(sha(physical.P4_PREPARE) == physical.P4_PREPARE_SHA and sha(physical.P4_OUTPUT/'report.json') == physical.P4_SHA, 'actual P4 ancestry authority')
    strict = rolling.consume(physical.P4_OUTPUT, physical.P4_SHA, physical.P4_PREPARE, physical.P4_PREPARE_SHA)
    require(strict['status'] == 'all8_rolling_saved_verified', 'complete existing P4 saved authority')
    layers = [deepcopy(x) for x in original_p['layers'] if x['arm'] == old.ARMS[0]]
    require([x['source_slot'] for x in layers] == [0, 1], 'fixed GN prefix for each known root, not outcome selected')
    events = physical.journal(physical.P4_OUTPUT/'journal.jsonl', old.CAP)
    ancestors = []
    for slot in (0, 1):
        eid = p4.episode_id(slot, old.ARMS[0], 'R')
        frames = [e['receipt'] for e in events if e['kind'] == 'fresh_observation' and e.get('episode_id') == eid
            and (e['receipt']['phase'] == 'warmup' or e['receipt']['step'] <= 5)]
        require(len(frames) == 17, 'original complete native prefix ancestry')
        for frame in frames:
            physical.png(frame, physical.P4_OUTPUT)
        ancestors.append(frames)
    bindings = dict(original_p['bindings'])
    for extra in (authority['sources'], {str(previous.TASK_AUTHORITY):previous.TASK_AUTHORITY_SHA}):
        for f, h in extra.items():
            require(f not in bindings or bindings[f] == h, 'consistent source inventory')
            bindings[f] = h
    files = [Path(__file__), ROOT/f'tests/{NAME}_test.py', ROOT/f'innerdocs/{NAME}.md',
        Path(paid.__file__), Path(original.__file__), Path(physical.__file__), Path(rolling.__file__),
        Path(previous.__file__), Path(p4.__file__), physical.P4_PREPARE, physical.P4_OUTPUT/'report.json',
        physical.P4_OUTPUT/'journal.jsonl']
    for frames in ancestors:
        bindings.update({f['path']:f['file_sha256'] for f in frames})
    bindings.update({str(f.resolve()):sha(f) for f in files})
    paid.contract.check_sources(bindings)
    result = dict(schema=NAME+'.prepare', protocol=protocol(), tasks=tasks, layers=layers,
        modelbinding=original_p['modelbinding'], original_manifest=original_p['manifest'], ancestors=ancestors,
        bindings=bindings, denominator=roster(), native_caps=native_caps(),
        arm_caps={m:arm_caps(m) for m in MODES}, P4_receipt_sha256=digest(strict), **FLAGS)
    load_data(result)  # Saved task/source pairing, no environment/model invocation.
    return result


def prepare(output):
    path = Path(output).absolute()
    require(path.parent == ROOT/'results' and path.name.startswith(NAME+'_'), 'exclusive own prepare prefix')
    p = build()
    require(len(canonical(p)) <= 16*1024**2 and shutil.disk_usage(path.parent).free >= FLOOR+len(canonical(p)), 'prepare cap/floor')
    with path.open('xb') as f:
        require(f.write(canonical(p)) == len(canonical(p)), 'complete prepare')
        f.flush()
        os.fsync(f.fileno())
    paid.base.fsync_dir(path.parent)
    return dict(prepared=str(path), prepare_sha256=sha(path))


def verify(path, expected):
    core.hash_value(expected)
    require(sha(path) == expected, 'actual external prepare SHA')
    p = read(path, 16*1024**2)
    require(canonical(p) == canonical(build()), 'complete source/protocol/native ancestry rebuild')
    return p


def execute(p, data, store, runtime, factory, final_check):
    rows, errors = roster(), []
    meter = Meter(store, runtime['check_limits'])
    store.put('roster.json', dict(rows=rows, protocol=protocol()))
    for row in rows:
        meter.episode, meter.region = row['episode_id'], 'setup'
        env = None
        try:
            env = meter.call('constructor', lambda: factory(meter))
            body = env.unwrapped
            require(body._live_meter is meter and body._live_constructor_complete is True and body.with_target is False
                and body.render_action is False and body.render_size == 224 and body.relative is True
                and body.action_scale == 100 and body.control_hz == 10 and body.dt == .01, 'unchanged actual native runtime')
            s = row['source_slot']
            row['status'] = 'running'
            episode(env, data['references'][s], data['plan'], p['layers'][s], data['tasks'][s], p['ancestors'][s],
                row, runtime, meter, store)
        except BaseException as exc:
            row.update(status='engineering_failure', first_error=repr(exc))
        finally:
            meter.region = 'closing'
            if env is not None:
                try:
                    meter.call('close', env.close)
                except BaseException as exc:
                    row['secondary_errors'].append(repr(exc))
                    row['status'] = 'engineering_failure'
            store.put(row['episode_id']+'_episode.json', row)
            store.emit('episode_closed', receipt=row)
    try:
        final = final_check()
        require(final['before'] == final['after'], 'whole-run model/buffer unchanged')
        paid.contract.check_sources(p['bindings'])
    except BaseException as exc:
        errors.append(repr(exc))
        final = None
    comparisons = []
    for s in (0, 1):
        pair = [r for r in rows if r['source_slot'] == s]
        complete = all(r['status'] == 'completed20' and r['full_horizon'] for r in pair)
        comparisons.append(dict(source_slot=s, same_realized_public20=complete,
            metrics20=[r.get('final_metrics') for r in pair] if complete else None,
            actual_public_steps=[r['control_returned'] for r in pair]))
    result = dict(schema=NAME+'.report', protocol=protocol(), rows=rows, fixed_denominator=4,
        comparisons=comparisons, counts=meter.counts, pending=meter.pending, errors=errors,
        model=final, internal_native_eval_calls=None, wall_seconds=time.monotonic()-meter.started,
        **FLAGS)
    result['status'] = 'known_calibration_recorded' if not errors and not meter.pending and all(
        r['status'] in ('completed20', 'refused_full_H_exceeds_remaining', 'refused_undefined_reconfiguration',
                        'native_stopped_warmup', 'native_stopped_control') for r in rows) else 'partial_or_failed'
    store.emit('run_finished', report_sha256=digest(result))
    result.update(journal_sha256=sha(store.path/'journal.jsonl'), journal_chain_tail=store.previous)
    store.write('report.json', canonical(result), final=True)
    return result


def run(prepared, expected, output, *, authorization):
    require(authorization == AUTH, 'explicit MAIN known-root CUDA/physics authorization before imports')
    started = time.monotonic()
    p = verify(prepared, expected)
    output = Path(output).absolute()
    require(output.parent == ROOT/'results' and output.name.startswith(NAME+'_'), 'new exclusive output')
    data = load_data(p)
    store = Store(output)
    store.put('execution_binding.json', dict(prepared=str(Path(prepared).absolute()), prepare_sha256=expected))
    store.put('startup_roster.json', dict(rows=roster()))
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host = require_cuda_allocation('cuda:0')
        device = torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        def check():
            require(time.monotonic()-started < WALL and torch.cuda.memory_allocated(device) <= 3*1024**3 and
                torch.cuda.memory_reserved(device) <= 4*1024**3, 'whole-job CUDA/wall caps')
            store.check()
        paths = {Path(f).parent for f in p['modelbinding']['model_files']}
        require(len(paths) == 1, 'single original checkpoint path')
        check()
        model = load_compatible_pretrained(paths.pop(), device=device).eval().requires_grad_(False)
        before = paid.base.model_hash(model)
        require(before['original_state_dict'] == p['modelbinding']['tensor_sha256'], 'actual original frozen state_dict')
        def final():
            check()
            after = paid.base.model_hash(model)
            require(all(not m.training for m in model.modules()) and all(not v.requires_grad for v in model.parameters()), 'frozen eval before/after')
            return dict(before=before, after=after, host=host, wall_seconds=time.monotonic()-started,
                CUDA_peak_bytes=[torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)])
        with torch.enable_grad(), torch.autocast(device_type='cuda', enabled=False), sdpa_kernel(SDPBackend.MATH):
            result = execute(p, data, store, dict(model=model, device=device, check_limits=check), old.native_factory, final)
        return dict(output=str(output), report_sha256=sha(output/'report.json'), status=result['status'])
    except BaseException as exc:
        store.write('uncaught_failure.json', canonical(dict(error=repr(exc), rows=roster(), **FLAGS)), final=True)
        raise


def events_by_group(events, group):
    return [e['payload'] for e in events if e['kind'] == 'model_trace' and e['group'] == group]


def consume(output, report_sha, prepared, prepare_sha):
    """Strict saved tensor, native trajectory, RGB, dispatch and budget closure."""
    output = Path(output).absolute()
    p = verify(prepared, prepare_sha)
    require(sha(output/'report.json') == report_sha, 'actual external terminal report SHA')
    report = read(output/'report.json', 256*1024**2)
    require(report['schema'] == NAME+'.report' and report['protocol'] == protocol() and report['fixed_denominator'] == 4, 'fixed new design')
    require(read(output/'execution_binding.json') == dict(prepared=str(Path(prepared).absolute()), prepare_sha256=prepare_sha), 'actual execution binding')
    require(sha(output/'journal.jsonl') == report['journal_sha256'], 'complete durable journal SHA')
    events = physical.journal(output/'journal.jsonl', CAP)
    require(events[-1]['kind'] == 'run_finished' and events[-1]['event_sha256'] == report['journal_chain_tail'], 'complete final journal')
    bare = {k:v for k,v in report.items() if k not in ('journal_sha256', 'journal_chain_tail')}
    require(events[-1]['report_sha256'] == digest(bare), 'journal/report exact join')
    require([r['episode_id'] for r in report['rows']] == [r['episode_id'] for r in roster()], 'all four failure denominators')
    # Preserve failed outputs, never release from their good subset.
    if report['status'] != 'known_calibration_recorded' or report['errors'] or report['pending']:
        return dict(schema=NAME+'.consume', status='partial_preserved_not_verified_complete',
            fixed_denominator=4, rows=report['rows'], report_sha256=report_sha, **FLAGS)
    require(report['model']['before'] == report['model']['after'] and
        report['model']['before']['original_state_dict'] == p['modelbinding']['tensor_sha256'], 'actual frozen weights/buffers')
    data = load_data(p)
    counts = {k:dict(attempted=0, completed=0) for k in native_caps()}
    pending, last_id = {}, -1
    for e in events:
        if e['kind'] not in ('operation_attempted', 'operation_completed'):
            continue
        entry = {k:e[k] for k in ('operation_id', 'episode_id', 'resource', 'units', 'region')}
        k, ident, units = entry['resource'], entry['operation_id'], entry['units']
        require(k in counts and type(units) is int and units > 0, 'known actual native/encoder axis')
        if e['kind'] == 'operation_attempted':
            require(type(ident) is int and ident > last_id, 'unique actual dispatch id')
            last_id = ident
            pending[ident] = entry
            counts[k]['attempted'] += units
        else:
            require(pending.pop(ident, None) == entry, 'matching native actual return')
            counts[k]['completed'] += units
    require(not pending and counts == report['counts'] and all(v['attempted'] == v['completed'] <= native_caps()[k] for k,v in counts.items()), 'full native census/caps')
    audited = []
    for row in report['rows']:
        eid, s = row['episode_id'], row['source_slot']
        require(read(output/(eid+'_episode.json'), 64*1024**2) == row, 'durable episode exact')
        frames = [e['receipt'] for e in events if e['kind'] == 'fresh_observation' and e['episode_id'] == eid]
        for frame in frames:
            physical.png(frame, output)
        if not row['initial_exact'] or row['control_returned'] < 5:
            require(row['status'] in ('native_stopped_warmup', 'native_stopped_control'), 'no false initial completion')
            continue
        original_frames = [f for f in frames if f['phase'] == 'warmup' or f['step'] <= 5]
        require(len(original_frames) == 17, 'all original prefix frames')
        for a, b in zip(original_frames, p['ancestors'][s], strict=True):
            physical.same_frame(a, b)
        context = deepcopy(p['tasks'][s]['root_context'])
        layer = p['layers'][s]
        turns = {r['boundary']:r for r in row['rounds']}
        accumulated = dict.fromkeys(core.AXES, 0)
        all_seals = []
        for encoding in row['encodings']:
            b = encoding['step']
            # Reuse actual P4 full encoding/alias/NPY/causality validator.
            z, observation = rolling.encoding(rolling.Reader(), output, row, b, layer,
                data['tasks'][s]['model_bindings'], p['modelbinding']['tensor_sha256'], events)
            raw10 = np.concatenate((data['references'][s]['raw_history'], np.asarray(row['executed_raw'][:b], np.float32)))[-10:]
            context = p4.roll(context, z, raw10, p['tasks'][s])
            if b == 20:
                require(row['status'] == 'completed20' and row['actual_Q20'] == core.model_cost(p['tasks'][s], context), 'terminal20 original Q')
                continue
            turn = turns[b]
            child = child_packet(eid, row['mode'], b, context, raw10, observation)
            signal = turn['signal']
            require(signal['child'] == child and read(output/(child['child_id']+'_signal.json'), 64*1024**2) == signal, 'own actual RGB/context/signal')
            acq = signal['acquisition']
            decide = signal['decision'] is not None
            sig = paid.validate_trace(p['tasks'][s], child, acq, events_by_group(events, acq['group']), int(decide))
            if decide:
                require(signal['decision'] == core.choose(p['tasks'][s], dict(context=context, depth=0), sig, row['mode']), 'unchanged natural scheduler')
            for k in core.AXES:
                accumulated[k] += acq['resource_receipt']['attempted'][k]
            program = turn['program']
            if program is not None:
                require(read(output/(child['child_id']+'_program.json'), 64*1024**2) == program, 'durable program')
                validate_program(p['tasks'][s], child, signal, program, events_by_group(events, child['child_id']+':program'))
                seal = turn['seal']
                require(read(output/(child['child_id']+'_dispatch.json')) == seal and seal['seal_sha256'] == digest({k:v for k,v in seal.items() if k != 'seal_sha256'}) and
                    seal['actions'] == program['selected']['actions'] and seal['program_sha256'] == digest(program), 'full-H sealed dispatch')
                all_seals.append(seal)
                for k in core.AXES:
                    accumulated[k] += program['resource_receipt']['attempted'][k]
        require(row['resources']['caps'] == arm_caps(row['mode']) and row['resources']['attempted'] == row['resources']['completed'] == accumulated and
            all(accumulated[k] <= arm_caps(row['mode'])[k] for k in core.AXES), 'complete independent actual arm costs')
        returns = [e for e in events if e['kind'] == 'control_return' and e['episode_id'] == eid]
        dispatch = [e for e in events if e['kind'] == 'authorized_dispatch' and e['episode_id'] == eid]
        require(len(returns) == len(dispatch) == row['control_returned'] and [e['step'] for e in returns] == list(range(1, row['control_returned']+1)) and
            [e['command'] for e in returns] == row['executed_raw'], 'all actual controls/returns no old result substitution')
        for a, b in zip(dispatch, returns, strict=True):
            require(a['index'] < b['index'] and a['step'] == b['step'] and a['command'] == b['command'], 'exact actual action order')
            if a['step'] <= 5:
                require(a['command'] == layer['commands'][a['step']-1], 'same original prefix')
            else:
                seal = next(x for x in all_seals if x['seal_sha256'] == a['seal_sha256'])
                require(a['command'] == seal['actions'][a['step']-seal['boundary']-1], 'full-H selected raw at actual ordinal')
                sealed = next(e for e in events if e['kind'] == 'full_program_sealed' and e['seal'] == seal)
                require(sealed['index'] < a['index'], 'seal before native dispatch')
        audited.append(dict(episode_id=eid, actual_public_steps=row['control_returned'], status=row['status'], model_axes=accumulated))
    return dict(schema=NAME+'.consume', status='all4_known_calibration_saved_verified', fixed_denominator=4,
        completed20=sum(r['full_horizon'] for r in report['rows']), rows=audited, counts=counts,
        comparisons=report['comparisons'], report_sha256=report_sha, source_sha256=sha(__file__),
        independent_neural_replay=False, **FLAGS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'run', 'consume'))
    for key in ('prepared', 'prepare-sha', 'output', 'report-sha'):
        parser.add_argument('--'+key)
    parser.add_argument('--authorize-known-cuda-and-physics', action='store_true')
    a = parser.parse_args()
    if a.command == 'prepare':
        result = prepare(a.output)
    elif a.command == 'verify':
        result = dict(schema=verify(a.prepared, a.prepare_sha)['schema'])
    elif a.command == 'consume':
        result = consume(a.output, a.report_sha, a.prepared, a.prepare_sha)
    else:
        require(a.authorize_known_cuda_and_physics, 'MAIN explicit authorization')
        result = run(a.prepared, a.prepare_sha, a.output, authorization=AUTH)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
