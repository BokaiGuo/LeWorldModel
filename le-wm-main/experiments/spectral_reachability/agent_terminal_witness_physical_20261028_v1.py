"""Native known2 x off/on raw10 terminal-witness adapter; MAIN physics only.

Own authority/schema; frozen mixedmode episode is reused, never its executor,
prepare, old receipts or globals. No inheritance or neural execution.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import io
import os
from pathlib import Path
import shutil
import time

import numpy as np
from experiments.spectral_reachability import agent_mixedmode_physical_20261028_v1 as mixed
from experiments.spectral_reachability import agent_p1_terminal_witness_selector_20261028_v1 as selector

NAME = 'agent_terminal_witness_physical_20261028_v1'
ROOT, ORDER = mixed.ROOT, mixed.ORDER
AUTH = 'MAIN_TERMINAL_WITNESS4_RAW10_PHYSICS_ONLY'
RECORDED = 'terminal_witness4_physical_recorded_not_admitted'
require, sha, raw, read = mixed.require, mixed.sha, mixed.raw, mixed.read
source, moving, paired = mixed.source, mixed.moving, mixed.paired
Meter = mixed.Meter
CAP, FLOOR, FINAL, WALL = mixed.CAP, mixed.FLOOR, mixed.FINAL, mixed.WALL
PREP_CAP = 64 * 1024**2
ANALYSIS = ROOT / 'results/agent_p1_terminal_witness_selector_20261028_v1_MAIN01.json'
ANALYSIS_SHA = 'a3a317b557da8bb6eb7a38d582f7df4d230b1b849cced00eca1add85403e63d9'
ARGUMENTS = {'selector_analysis', 'selector_analysis_sha256'}


def contract():
    c = deepcopy(mixed.allmode.contract())
    c.update(schema=NAME + '.contract', arm='terminal_witness_mixed222',
        producer=selector.NAME + '.analysis + exact selector.build reconstruction; no selector prepare/consume exists',
        raw_authority='exact newly consumed terminal-witness raw10, no tanh/clip/scale/reselection',
        prefix51='own warmup0..10 + own plan0..5, actual off/on/off triplets only',
        inheritance=False, freshness=False, prepare_cap_bytes=PREP_CAP,
        objective='minimum saved nonlinear terminal Q over full fixed222, stable index tie',
        no_outcome_selection=True)
    return c


class Store(mixed.Store):
    """Original flat durable methods/caps with only NEW exclusive output name."""
    def __init__(self, path):
        self.path = Path(mixed.absolute(str(path)))
        require(self.path.name.startswith(NAME + '_') and not self.path.exists() and
                not self.path.is_symlink() and self.path.parent.is_dir() and
                shutil.disk_usage(self.path.parent).free >= FLOOR + CAP, 'NEW native output32MiB plus200MiB floor')
        self.path.mkdir(); paired.fsync_dir(self.path.parent); self.previous = None; self.index = 0


def validate_execution(p):
    require(p['schema'] == NAME + '.prepare' and p['contract'] == contract() and
            p['source_report_sha256'] == mixed.SOURCE_SHA and len(p['plans']) == 4 and
            len(p['source_roots']) == len(p['source_slots']) == len(p['warm_traces']) == 2,
            'own terminal authority/full4 source preparation')
    authority = p['selector_authority']
    require(authority['new_algorithm'] is True and authority['selection_uses_physics'] is False and
            authority['P1_admission'] is False and authority['plans_sha256'] == mixed.core.digest(p['plans']),
            'new selector authority and unchanged complete plan payload')
    mixed.actual_hash(authority['analysis_sha256'])
    require(authority['analysis_sha256'] == ANALYSIS_SHA and authority['analysis'] == str(ANALYSIS) and
            authority['selector'] == selector.NAME and authority['exact_build_verified'] is True and authority['family'] == 'mixedmode',
            'real pinned analyze/build authority, no invented prepare or consume receipt')
    for path, expected in authority['source_sha256'].items():
        mixed.absolute(path); mixed.actual_hash(expected)
    for i, plan in enumerate(p['plans']):
        require(type(plan['root_slot']) is int and (plan['root_slot'], plan['condition']) == ORDER[i] and
                type(plan['selected_index']) is int and 0 <= plan['selected_index'] < 222 and
                plan['arm'] == contract()['arm'] and plan['selection_uses_physics'] is False and
                plan['P1_admission'] is False, 'ordered all4 terminal-selected raw10, no old arm authority')
        require(plan['schema'] == NAME + '.execution_plan' and plan['selector_analysis_sha256'] == ANALYSIS_SHA and
                plan['source_report_sha256'] == mixed.SOURCE_SHA and plan['raw_steps'] == 10 and
                plan['selection_digest'] == mixed.core.digest({k: v for k, v in plan.items() if k != 'selection_digest'}),
                'own newplan schema/digest preserves terminal-witness authority')
        commands = mixed.raw10(plan['commands'])
        require(hashlib.sha256(commands.tobytes()).hexdigest() == plan['raw_float32_sha256'], 'literal selected raw10 bits')


def own_files():
    return [Path(__file__).resolve(), ROOT / 'tests' / ('test_' + NAME + '.py'), ROOT / 'innerdocs' / (NAME + '.md')]


def plans_from_analysis(analysis, compiled, data):
    require(analysis['schema'] == selector.NAME + '.analysis' and analysis['new_algorithm'] is True and
            analysis['P1_admission'] is analysis['independent_confirmation'] is False and
            analysis['status'] == 'new_exploratory_model_only_selector_not_admitted', 'actual exploratory selector schema/flags')
    panels = [p for p in analysis['panels'] if p['family'] == 'mixedmode']
    require(len(panels) == len(compiled['slots']) == 4 and
            compiled['schema'] == mixed.PRODUCER + '.prepare' and compiled['source_report_sha256'] == mixed.SOURCE_SHA,
            'mixed222 ONLY fixed four tasks, no family selected using scores/outcomes')
    plans = []
    for i, (panel, slot) in enumerate(zip(panels, compiled['slots'], strict=True)):
        root, condition = ORDER[i]
        require(type(panel['index']) is type(panel['root_slot']) is int and panel['index'] == slot['index'] == i and
                (panel['root_slot'], panel['condition']) == (slot['root_slot'], slot['condition']) == ORDER[i] and
                panel['candidate_denominator'] == panel['terminal_available'] == len(panel['candidates']) == 222 and
                panel['literal_forward_requests'] == [] and panel['status'] == 'complete_model_only' and
                panel['P1_admission'] is False, 'all222 real terminal witnesses, fixed4 no subset or missing fallback')
        require(panel['task_sha256'] == slot['task_sha256'] and slot['blind_sha256'] == data['exports'][ORDER[i]]['sha256'] and
                panel['context_sha256'] == selector.sha_object(slot['packet']['root_context']) and
                panel['goal_sha256'] == selector.sha_object(slot['packet']['goal']) and
                panel['q_sha256'] == selector.base.digest(slot['q']), 'original condition-specific task/context/goal/Q identity')
        candidates = panel['candidates']
        require([c['index'] for c in candidates] == list(range(222)) and all(
                type(c['index']) is int and type(c['nonlinear_terminal_Q']) in (int, float) and
                np.isfinite(c['nonlinear_terminal_Q']) and c['nonlinear_terminal_Q'] >= 0 for c in candidates),
                'full model-only terminal objective denominator')
        winner = min(candidates, key=lambda c: (c['nonlinear_terminal_Q'], c['index']))
        selected = panel['nonlinear_terminal_Q']; commands = mixed.raw10(selected['commands'])
        require(selected['index'] == winner['index'] and selected['score'] == winner['nonlinear_terminal_Q'] and
                hashlib.sha256(commands.tobytes()).hexdigest() == selected['raw_float32_sha256'] == winner['raw_float32_sha256'] and
                selected['model_prefix_uphill_from_anchor'] is winner['model_prefix_uphill_from_anchor'],
                'stable full222 terminal argmin, original raw witness never regenerated')
        plan = dict(schema=NAME + '.execution_plan', index=i, root_slot=root, source_slot=root, condition=condition,
            arm=contract()['arm'], task_id=slot['task_id'], task_sha256=slot['task_sha256'], blind_sha256=slot['blind_sha256'],
            source_report_sha256=mixed.SOURCE_SHA, selector_analysis_sha256=ANALYSIS_SHA, selected_index=winner['index'],
            commands=deepcopy(selected['commands']), raw_float32_sha256=selected['raw_float32_sha256'], raw_steps=10,
            nonlinear_terminal_Q=selected['score'], model_prefix_uphill_from_anchor=selected['model_prefix_uphill_from_anchor'],
            context_sha256=panel['context_sha256'], goal_sha256=panel['goal_sha256'], q_sha256=panel['q_sha256'],
            original_stage_plus_ridge_index=panel['original_stage_plus_ridge']['index'],
            selection_uses_physics=False, P1_admission=False)
        plan['selection_digest'] = mixed.core.digest(plan); plans.append(plan)
    return plans


def material(arguments):
    require(type(arguments) is dict and set(arguments) == ARGUMENTS, 'explicit actual selector analyze authority')
    path = Path(mixed.absolute(arguments['selector_analysis']))
    require(path == ANALYSIS and arguments['selector_analysis_sha256'] == ANALYSIS_SHA == sha(path),
            'pinned actual selector analysis SHA, no future/default certificate')
    analysis = read(path, 8 * 1024**2)
    for file, expected in analysis['source_sha256'].items(): require(sha(file) == expected, 'actual selector source/tests/doc closure')
    require(analysis == selector.build() and sha(path) == ANALYSIS_SHA, 'exact entire real selector.build reconstruction')
    # The selection has already completed using model-only saved evidence;
    # native source data are read below solely to bind/recreate the fixed tasks.
    upstream = next(x for x in analysis['inputs'] if x['family'] == 'mixedmode')
    require(upstream['prepare_sha256'] == mixed.PREP_SHA and upstream['report_sha256'] == mixed.REPORT_SHA and
            sha(upstream['prepared']) == upstream['prepare_sha256'] and sha(upstream['report']) == upstream['report_sha256'],
            'original full mixed222 model evidence identity')
    compiled = read(upstream['prepared'], selector.mixed.PREP_CAP)
    data = mixed.source_material(); plans = plans_from_analysis(analysis, compiled, data)
    bindings = {}; mixed.merge(bindings, compiled['bindings']); mixed.merge(bindings, data['bindings'])
    mixed.merge(bindings, analysis['source_sha256'])
    mixed.merge(bindings, {str(path): ANALYSIS_SHA, upstream['prepared']: upstream['prepare_sha256'],
        upstream['report']: upstream['report_sha256']})
    for item in analysis['inputs']:
        mixed.merge(bindings, {item['prepared']: item['prepare_sha256'], item['report']: item['report_sha256']})
    for slot in compiled['slots']: mixed.merge(bindings, {slot['task_file']: slot['task_sha256']})
    for file in [*own_files(), Path(mixed.__file__).resolve(), Path(mixed.allmode.__file__).resolve()]:
        mixed.merge(bindings, {str(file): sha(file)})
    native_path, native_sha = mixed.allmode.constructor_contract()
    mixed.merge(bindings, {native_path: native_sha})
    authority = dict(selector=selector.NAME, analysis=str(path), analysis_sha256=ANALYSIS_SHA,
        source_sha256=deepcopy(analysis['source_sha256']), exact_build_verified=True, family='mixedmode',
        new_algorithm=True, selection_uses_physics=False, P1_admission=False,
        plans_sha256=mixed.core.digest(plans), actual_scope='analyze + verify/build; no selector prepare/consume exists')
    p = dict(schema=NAME + '.prepare', contract=contract(), arguments=deepcopy(arguments), plans=plans,
        selector_authority=authority, bindings=bindings, source_report_sha256=mixed.SOURCE_SHA,
        source_slots=[deepcopy(r['slot']) for r in data['roots']],
        warm_traces=[deepcopy(r['warm_trace']) for r in data['roots']], source_roots=mixed.allmode.root_metadata(data))
    validate_execution(p)
    for file, expected in bindings.items(): require(sha(file) == expected, 'final source/metadata hash closure')
    require(len(raw(p)) <= PREP_CAP, 'complete prepare metadata cap, no binding truncation')
    return p, data


def write_new(path, value, cap):
    path = Path(mixed.absolute(str(path))); payload = raw(value)
    require(path.parent == ROOT / 'results' and path.name.startswith(NAME + '_') and not path.exists() and
            not path.is_symlink() and len(payload) <= cap and shutil.disk_usage(path.parent).free >= FLOOR + len(payload),
            'NEW own bounded metadata, no overwrite')
    with path.open('xb') as f:
        require(f.write(payload) == len(payload), 'full metadata write'); f.flush(); os.fsync(f.fileno())
    paired.fsync_dir(path.parent)
    return dict(output=str(path), sha256=sha(path), bytes=len(payload))


def prepare(output, arguments):
    path = Path(mixed.absolute(str(output))); require(not path.exists(), 'exclusive prepare before expensive build')
    p, _ = material(arguments); return write_new(path, p, PREP_CAP)


def verify(prepared, expected):
    mixed.actual_hash(expected); mixed.absolute(str(prepared))
    require(sha(prepared) == expected, 'external native prepare SHA')
    p = read(prepared, PREP_CAP); now, data = material(p['arguments'])
    require(p == now and sha(prepared) == expected, 'whole deterministic selector/source/plan prepare reconstruction')
    return p, data


def rows_for(p):
    return [dict(index=i, root_slot=plan['root_slot'], condition=plan['condition'], arm=plan['arm'],
        task_id=plan['task_id'], selection_digest=plan['selection_digest'], plan=deepcopy(plan),
        status='not_started', first_error=None, secondary_errors=[], warmup_attempted=0, warmup_returned=0,
        own_attempted=0, own_returned=0, PNG=0, initial_exact=False, trajectory=[], checkpoints=[],
        missing_checkpoints=[0, 5, 10], last_observation=None, responses=[], final_state=None,
        final_metrics=None, native_success=None, terminated=None, truncated=None,
        prefix51=dict(complete=False, frames=[]), source_unpaired_on_steps=[],
        evidence_origin='new_native_execution_known_root') for i, plan in enumerate(p['plans'])]


def execute(p, data, store, factory, *, authorization, prepared, prepare_sha256, started=None):
    require(authorization == AUTH, 'MAIN explicit physical authority before callbacks')
    mixed.actual_hash(prepare_sha256); validate_execution(p)
    require(p['source_slots'] == [r['slot'] for r in data['roots']] and
            p['source_roots'] == mixed.allmode.root_metadata(data) and
            p['warm_traces'] == [r['warm_trace'] for r in data['roots']], 'actual source identity/warmup history')
    meter = Meter(store, started); rows = rows_for(p); errors = []; seen = []
    binding = dict(prepared=mixed.absolute(str(prepared)), prepare_sha256=prepare_sha256,
        source_report_sha256=mixed.SOURCE_SHA, selector_authority=deepcopy(p['selector_authority']))
    def fail(stage, exc, row=None):
        item = mixed.base.error(stage, exc); errors.append(item)
        if row is not None:
            if row['first_error'] is None: row['first_error'] = deepcopy(item)
            else: row['secondary_errors'].append(deepcopy(item))
            row['status'] = 'engineering_failed'
    try:
        store.put('execution_binding.json', binding)
        store.emit('run_started', prepare_sha256=prepare_sha256, contract=contract())
        for row, plan in zip(rows, p['plans'], strict=True):
            meter.where = dict(branch=row['index'], **{k: row[k] for k in
                ('root_slot', 'condition', 'arm', 'selection_digest')})
            env = None; row['status'] = 'running'
            try:
                env = meter.call('constructor', factory)
                require(all(env is not prior and env.unwrapped is not prior.unwrapped for prior in seen),
                        'fresh native env per arm'); seen.append(env)
                mixed.validate_env(env.unwrapped)
                require(env.unwrapped.render_action is False and env.unwrapped.render_size == 224 and
                        env.unwrapped.render_mode == 'rgb_array', 'original renderer/raw control semantics')
                with mixed.native_hooks(env, meter, row['secondary_errors']):
                    mixed.episode(env, data['roots'][plan['root_slot']], data['source_plan'], plan, row, meter, store)
                require(not row['secondary_errors'], 'native hooks restored')
            except mixed.allmode.NativeStop as exc:
                row.update(status='native_stopped', stop_reason=str(exc))
            except BaseException as exc:
                fail('branch', exc, row)
            finally:
                if env is not None:
                    try: meter.call('close', env.close, cleanup=True)
                    except BaseException as exc: fail('close', exc, row)
                if row['secondary_errors'] and row['first_error'] is None:
                    fail('cleanup', ValueError('native cleanup secondary error'), row)
                for phase, label in (('warmup', 'warmup'), ('plan', 'own')):
                    ops = [op for op in meter.receipts if op['resource'] == 'public_step' and
                           op.get('branch') == row['index'] and op['phase'] == phase]
                    row[label + '_attempted'] = len(ops)
                    row[label + '_returned'] = sum(op['status'] == 'returned' for op in ops)
            store.emit('branch_closed', receipt=row)
    except BaseException as exc:
        fail('execution', exc)
    source_ok = False
    try:
        for file, expected in p['bindings'].items(): require(sha(file) == expected, 'final full input/source closure')
        source_ok = True
    except BaseException as exc:
        fail('final_inputs', exc)
    errors.extend(meter.secondary)
    if time.monotonic() - meter.started >= WALL: fail('final_wall', ValueError('900s wall'))
    good = not errors and source_ok and all(r['status'] in ('horizon_completed', 'native_stopped') for r in rows)
    status = RECORDED if good else 'partial_or_failed'
    try: store.emit('run_finished', status=status, counts=meter.counts, errors=errors)
    except BaseException as exc: fail('final_journal', exc); status = 'partial_or_failed'
    report = dict(schema=NAME + '.report', status=status, contract=contract(), execution_binding=binding,
        rows=rows, errors=errors, counts=deepcopy(meter.counts), actual_operation_receipts=deepcopy(meter.receipts),
        fixed_denominator=4, known_roots=2, fresh_roots=0, inherited=0, inputs_final_verified=source_ok,
        new_neural_calls=0, new_independent_samples=0, P1_admission=False, P3_admission=False,
        scientific_admission=False, freshness=False, equal_total_compute=False, equal_FLOPs=False,
        training=False, independent_confirmation=False,
        total_internal_physics_scope='metered_native_calls' if meter.counts['constructor']['attempted'] ==
            meter.counts['constructor']['completed'] else 'unknown_constructor_work',
        journal_sha256=None, execution_binding_sha256=None, journal_chain_tail=store.previous,
        wall_seconds=time.monotonic() - meter.started)
    for name, key in (('journal.jsonl', 'journal_sha256'), ('execution_binding.json', 'execution_binding_sha256')):
        try: report[key] = sha(store.path / name)
        except BaseException as exc: fail('final_' + key, exc); report['status'] = 'partial_or_failed'
    try: store.put('report.json', report, final=True)
    except BaseException as exc: fail('final_report', exc); report['status'] = 'partial_or_failed'
    return report


def validate_saved(r, events, p):
    """Own-schema native trace validator, never a projected mixedmode report."""
    validate_execution(p)
    require(r['schema'] == NAME + '.report' and r['contract'] == contract() and
            r['status'] in (RECORDED, 'partial_or_failed') and r['fixed_denominator'] == len(r['rows']) == 4 and
            r['known_roots'] == 2 and r['fresh_roots'] == r['inherited'] == 0 and
            r['execution_binding']['selector_authority'] == p['selector_authority'] and
            r['execution_binding']['source_report_sha256'] == mixed.SOURCE_SHA, 'own full4 terminal physical report authority')
    require(all(r[k] is False for k in ('P1_admission', 'P3_admission', 'scientific_admission', 'freshness',
            'equal_total_compute', 'equal_FLOPs', 'training', 'independent_confirmation')) and
            r['new_neural_calls'] == r['new_independent_samples'] == 0, 'known-root engineering only')
    require(events and events[0]['kind'] == 'run_started' and events[0]['contract'] == contract() and
            events[0]['prepare_sha256'] == r['execution_binding']['prepare_sha256'] and
            events[-1]['kind'] == 'run_finished' and all(events[-1][k] == r[k] for k in ('status', 'counts', 'errors')),
            'actual terminal bookends')
    allowed = {'run_started', 'run_finished', 'operation_dispatch_intent', 'operation_returned', 'operation_failed',
        'placement_recipe', 'goal_attached', 'invalid_observation', 'fresh_observation', 'returned_transition',
        'own_response', 'branch_closed'}
    tail = None; intents = {}; terminals = {}; stack = []; closed = 0
    for i, event in enumerate(events):
        require(event['kind'] in allowed and type(event['index']) is int and event['index'] == i and
                event['previous_sha256'] == tail and event['event_sha256'] ==
                source.digest({k: v for k, v in event.items() if k != 'event_sha256'}), 'known physical event/hash chain')
        tail = event['event_sha256']
        if event['kind'] == 'branch_closed':
            require(closed < 4 and event['receipt'] == r['rows'][closed], 'ordered exact full branch closure')
            closed += 1
        if 'branch' in event:
            require(type(event['branch']) is int and event['branch'] == closed and closed < 4 and
                    all(event[k] == r['rows'][closed][k] for k in ('root_slot', 'condition', 'arm', 'selection_digest')),
                    'active exact branch identity')
        if event['kind'] not in ('operation_dispatch_intent', 'operation_returned', 'operation_failed'): continue
        op = event['receipt']; n = op['index']; b = op['branch']
        require(type(n) is type(b) is int and b == closed and 0 <= b < 4 and op['resource'] in contract()['caps'] and
                all(op[k] == r['rows'][b][k] for k in ('root_slot', 'condition', 'arm', 'selection_digest')),
                'native resource exact selected branch')
        if event['kind'] == 'operation_dispatch_intent':
            require(n == len(intents) and op['status'] == 'pending', 'ordered actual operation intent')
            intents[n] = op; stack.append(n)
        else:
            require(stack and stack.pop() == n and n in intents and n not in terminals and
                    op['status'] == ('returned' if event['kind'] == 'operation_returned' else 'failed') and
                    dict({k: v for k, v in op.items() if k != 'error'}, status='pending') == intents[n],
                    'nested native return/failure exact closure')
            terminals[n] = op
    require(not stack and len(intents) == len(terminals) and tail == r['journal_chain_tail'] and
            r['actual_operation_receipts'] == [terminals[i] for i in range(len(terminals))] and
            [e['receipt'] for e in events if e['kind'] == 'branch_closed'] ==
            [row for row in r['rows'] if row['status'] != 'not_started'], 'complete attempted resource/failure denominator')
    counts = {k: dict(attempted=0, completed=0) for k in contract()['caps']}
    for op in r['actual_operation_receipts']:
        counts[op['resource']]['attempted'] += 1
        counts[op['resource']]['completed'] += op['status'] == 'returned'
    require(counts == r['counts'] and all(c['attempted'] <= contract()['caps'][k] for k, c in counts.items()),
            'all-resource independent actual counts/caps')
    for i, row in enumerate(r['rows']):
        plan = p['plans'][i]; task = p['source_roots'][plan['root_slot']]
        require(type(row['index']) is int and row['index'] == i and row['plan'] == plan and
                (row['root_slot'], row['condition']) == ORDER[i] and
                all(row[k] == plan[k] for k in ('arm', 'selection_digest', 'task_id')) and
                row['evidence_origin'] == 'new_native_execution_known_root' and
                row['status'] in ('not_started', 'engineering_failed', 'native_stopped', 'horizon_completed'),
                'all4 new physical rows, no old schema/plan/row inheritance')
        scoped = [e for e in events if e.get('branch') == i]
        ops = [op for op in r['actual_operation_receipts'] if op['branch'] == i]
        public = [op for op in ops if op['resource'] == 'public_step']
        transitions = [e for e in scoped if e['kind'] == 'returned_transition']
        observations = [e for e in scoped if e['kind'] == 'fresh_observation']
        for phase, label, commands in (('warmup', 'warmup', moving.warmup_commands()),
                                       ('plan', 'own', mixed.raw10(plan['commands']))):
            current = [op for op in public if op['phase'] == phase]
            require(len(current) <= 10 and [op['step'] for op in current] == list(range(1, len(current) + 1)) and
                    all(op['raw_bits'] == commands[j].tobytes().hex() for j, op in enumerate(current)) and
                    row[label + '_attempted'] == len(current) and
                    row[label + '_returned'] == sum(op['status'] == 'returned' for op in current),
                    'complete contiguous sealed raw10 authority, no changed actions or padding')
            returned = [t for t in transitions if t['phase'] == phase]
            require([t['step'] for t in returned] == list(range(1, len(returned) + 1)) and
                    len(returned) <= row[label + '_returned'], 'actual returned transition chronology')
            for t in returned:
                require(t['actual_dtype'] == np.dtype(np.float32).str and
                        t['actual_bits'] == t['requested_bits'] == commands[t['step'] - 1].tobytes().hex(),
                        'actual raw float32 return bits')
        require(row['PNG'] == len(observations), 'all actual rendered observations counted')
        bridge = [{k: deepcopy(e[k]) for k in ('receipt', 'pre', 'post')} for e in observations
                  if e['receipt']['phase'] == 'warmup' or e['receipt']['step'] <= 5]
        require(row['prefix51'] == dict(complete=len(bridge) == 51, frames=bridge), 'actual prefix51, no artificial splicing')
        unknown = []; selected = []; wanted_overlay = 1 if row['condition'] == 'on' else 0
        for event in observations:
            f = event['receipt']; phase, step, overlay = f['phase'], f['step'], f['overlay_index']
            require(phase in ('warmup', 'plan') and type(step) is type(overlay) is int and
                    0 <= step <= 10 and overlay in (0, 1, 2) and f['condition'] == ('on' if overlay == 1 else 'off') and
                    event['pre'] == event['post'] and f['snapshot'] == event['post']['snapshot'] and
                    f['state'] == f['snapshot']['state'] and len(f['state']) == 7 and
                    event['pre']['renderer'] == task['renderer'] and event['pre']['goal_pose'] == task['goal_pose'],
                    'native7 observation/renderer/goalpose identity')
            if overlay == wanted_overlay: selected.append(f)
            if phase == 'warmup':
                ref = mixed.allmode.reference_for(p['warm_traces'][plan['root_slot']], step, overlay)
                if ref is None: unknown.append(step)
                else: mixed.allmode.check_warm_frame(event, ref)
            else:
                require(event['pre']['goal_state'] == task['goal'] and
                        event['pre']['goal_rgb_sha256'] == task['goal_pixels'][row['condition']],
                        'same actual task/goal, condition applied to observed history and goalRGB')
                if step:
                    t = next(t for t in transitions if t['phase'] == 'plan' and t['step'] == step)
                    require(f['state'] == t['state'] and f['snapshot'] == t['snapshot'], 'RGB/native dispatch state identity')
        require(row['source_unpaired_on_steps'] == unknown and row['trajectory'] == selected,
                'no invented historical on evidence; actual selected-condition whole trajectory')
        if row['initial_exact']:
            require(all(row[k] == task[k] for k in ('root_state', 'root_sha256', 'goal', 'goal_pose')) and
                    row['goal_pixel_sha256'] == task['goal_pixels'][row['condition']] and
                    source.root_hash(source.vector(row['root_state']), source.vector(row['goal'])) == row['root_sha256'],
                    'exact original native7 root and goal identities')
            goal_events = [e for e in scoped if e['kind'] == 'goal_attached']
            require(len(goal_events) == 1 and all(goal_events[0][k] == row[k] for k in ('goal', 'goal_pose')) and
                    goal_events[0]['goal_rgb_sha256'] == row['goal_pixel_sha256'], 'single original goal bind receipt')
        responses = row['responses']
        require([x['step'] for x in responses] == list(range(1, len(responses) + 1)) and
                len(responses) <= row['own_returned'] and
                [e['response'] for e in scoped if e['kind'] == 'own_response'] == responses, 'all actual own responses')
        for response in responses:
            calc = mixed.metrics(row['goal'], response['state'])
            delta = source.vector(response['state']) - source.vector(row['root_state'])
            require(calc['native'] == response['native_metrics'] and calc['native7_reward'] == response['reward'] and
                    response['native_success'] is response['terminated'] is calc['native_success'] and
                    type(response['truncated']) is bool and np.array_equal(delta, response['delta_native7']) and
                    response['delta_angle_wrapped'] == float(np.arctan2(np.sin(delta[4]), np.cos(delta[4]))),
                    'unchanged native7 reward and jointXY/angle success, wrapped angle diagnostic only')
            t = next(t for t in transitions if t['phase'] == 'plan' and t['step'] == response['step'])
            require(all(t[k] == response[k] for k in ('state', 'reward', 'terminated', 'truncated')) and
                    t['goal_state'] == row['goal'], 'actual dispatch response identity')
        if responses:
            last = responses[-1]
            require(row['final_state'] == last['state'] and row['final_metrics'] == last['native_metrics'] and
                    all(row[k] is last[k] for k in ('native_success', 'terminated', 'truncated')) and
                    all(not x['terminated'] and not x['truncated'] for x in responses[:-1]), 'actual native-stop/final endpoint')
        else:
            require(row['final_state'] is row['final_metrics'] is row['native_success'] is None, 'missing outcomes unknown, not zero')
        cp = [f for f in selected if f['phase'] == 'plan' and f['step'] in (0, 5, 10)]
        require(row['checkpoints'] == [{k: f[k] for k in ('step', 'png', 'file_sha256', 'pixel_sha256', 'state', 'snapshot')}
                for f in cp] and row['missing_checkpoints'] == [b for b in (0, 5, 10) if b not in [f['step'] for f in cp]],
                'checkpoint RGB/native7 identity and explicit missing denominator')
        if row['status'] in ('horizon_completed', 'native_stopped'):
            require(row['first_error'] is None and row['secondary_errors'] == [], 'no engineering error hidden as native stop')
            framekeys = [('warmup', b, k) for b in range(row['warmup_returned'] + 1) for k in (0, 1, 2)]
            if row['initial_exact']: framekeys += [('plan', b, k) for b in range(row['own_returned'] + 1) for k in (0, 1, 2)]
            require([(e['receipt']['phase'], e['receipt']['step'], e['receipt']['overlay_index']) for e in observations] ==
                    framekeys, 'all unpadded real RGB triplets')
            for j in range(0, len(observations), 3):
                a, b, c = observations[j:j + 3]
                require(a['receipt']['file_sha256'] == c['receipt']['file_sha256'] and
                        a['receipt']['pixel_sha256'] == c['receipt']['pixel_sha256'] and
                        a['receipt']['state'] == b['receipt']['state'] == c['receipt']['state'], 'off/on/off unchanged physics')
            if responses:
                require(row['warmup_returned'] == 10 and row['initial_exact'] and row['own_returned'] == len(responses) and
                        ((row['status'] == 'native_stopped') == (last['terminated'] or last['truncated'])), 'native stop never padded')
            if row['status'] == 'horizon_completed': require(row['own_returned'] == 10, 'actual full10, not only first5')
    if r['status'] == RECORDED:
        require(closed == 4 and r['errors'] == [] and r['inputs_final_verified'] is True and
                0 <= r['wall_seconds'] < WALL and all(x['status'] in ('horizon_completed', 'native_stopped') for x in r['rows']),
                'complete engineering four denominator only')
        c = {k: v['completed'] for k, v in counts.items()}
        require(all(v['attempted'] == v['completed'] for v in counts.values()) and
                c['constructor'] == c['close'] == 4 and c['reset'] == 8 and
                c['goal_bind'] == sum(x['initial_exact'] for x in r['rows']) and
                c['set_state'] == 2 * c['reset'] and c['goal_render'] == c['reset'] and
                c['goal_setter'] == c['reset'] + c['goal_bind'] and
                c['physics_step'] == 2 * c['reset'] + 10 * c['public_step'] and
                c['public_step'] == sum(x['warmup_returned'] + x['own_returned'] for x in r['rows']) and
                c['explicit_render'] == sum(x['PNG'] for x in r['rows']), 'independent full native physical census')
    return deepcopy(r['rows'])


def run(prepared, expected, output, factory, *, authorization=None):
    require(authorization == AUTH, 'MAIN authority before input reads/output/env')
    p, data = verify(prepared, expected); store = Store(output)
    return execute(p, data, store, factory, authorization=authorization, prepared=prepared, prepare_sha256=expected)


def consume(output, expected):
    mixed.actual_hash(expected); directory = Path(mixed.absolute(str(output)))
    files = mixed.saved_artifacts(directory)
    require(hashlib.sha256(files['report.json']).hexdigest() == expected, 'actual external terminal physical report SHA')
    r = mixed.decode(files['report.json']); binding = r['execution_binding']
    require(hashlib.sha256(files['execution_binding.json']).hexdigest() == r['execution_binding_sha256'] and
            mixed.decode(files['execution_binding.json']) == binding, 'actual execution authority binding')
    p, _ = verify(binding['prepared'], binding['prepare_sha256'])
    journal = files['journal.jsonl']
    require(journal.endswith(b'\n') and hashlib.sha256(journal).hexdigest() == r['journal_sha256'], 'saved physical journal SHA')
    events = [mixed.decode(line) for line in journal.splitlines()]
    rows = validate_saved(r, events, p); names = {'report.json', 'execution_binding.json', 'journal.jsonl'}
    from PIL import Image
    png_count = 0
    for event in events:
        if event['kind'] == 'invalid_observation':
            name = event['file']; require(Path(name).name == name and name not in ('', '.', '..') and
                hashlib.sha256(files[name]).hexdigest() == event['file_sha256'], 'retained invalid observation incident')
            names.add(name)
        if event['kind'] != 'fresh_observation': continue
        f = event['receipt']; name = f['png']
        require(Path(name).name == name and name not in ('', '.', '..') and name not in names and
                len(files[name]) <= 1024**2 and hashlib.sha256(files[name]).hexdigest() == f['file_sha256'],
                'unique own flat PNG bytes/SHA')
        with Image.open(io.BytesIO(files[name])) as image:
            require(image.format == 'PNG' and image.mode == 'RGB' and image.size == (224, 224), 'actual native RGB224')
            rgb = np.asarray(image).copy()
        require(moving.image_sha(rgb) == f['pixel_sha256'], 'saved actual RGB pixels')
        moving.guard(rgb, source.vector(f['state']))
        names.add(name); png_count += 1
    require(set(files) == names, 'no unaccounted output artifacts')
    bindings = deepcopy(p['bindings']); mixed.merge(bindings, {binding['prepared']: binding['prepare_sha256']})
    mixed.merge(bindings, {str(directory / name): hashlib.sha256(body).hexdigest() for name, body in files.items()})
    for file, h in p['bindings'].items(): require(sha(file) == h, 'final original input/source seals')
    require(mixed.saved_artifacts(directory) == files, 'saved artifact bytes unchanged')
    return dict(schema=NAME + '.consume', status=r['status'], rows=rows, counts=r['counts'], errors=r['errors'],
        fixed_denominator=4, known_roots=2, fresh_roots=0, inherited=0, saved_PNG_verified=png_count,
        physical_report_sha256=expected, selector_authority=p['selector_authority'], bindings=bindings,
        metric_breakdown=[dict(root_slot=row['root_slot'], condition=row['condition'],
            metrics=None if row['final_state'] is None else mixed.metrics(row['goal'], row['final_state'])) for row in rows],
        P1_admission=False, P3_admission=False, scientific_admission=False, freshness=False,
        equal_total_compute=False, equal_FLOPs=False, independent_confirmation=False, new_neural_calls=0, new_physics_calls=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'run', 'consume'))
    for key in ('output', 'prepared', 'prepare-sha', 'report-sha', 'analysis', 'analysis-sha', 'receipt', 'authorization'):
        parser.add_argument('--' + key)
    a = parser.parse_args(argv)
    if a.command == 'prepare':
        result = prepare(a.output, dict(selector_analysis=a.analysis, selector_analysis_sha256=a.analysis_sha))
    elif a.command == 'verify':
        p, _ = verify(a.prepared, a.prepare_sha)
        result = dict(status='verified_no_execution', selected_indices=[x['selected_index'] for x in p['plans']],
                      fixed_denominator=4, P1_admission=False)
    elif a.command == 'consume':
        receipt = consume(a.output, a.report_sha)
        result = write_new(a.receipt, receipt, PREP_CAP) if a.receipt else receipt
    else:
        require(a.authorization == AUTH, 'MAIN CLI authority before native imports')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401; original native registration only
            return gym.make('swm/PushT-v1', render_mode='rgb_array', resolution=224, with_target=False)
        result = run(a.prepared, a.prepare_sha, a.output, factory, authorization=a.authorization)
    print(source.canonical(result))
    return 2 if result.get('status') == 'partial_or_failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
