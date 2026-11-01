"""MAIN-only physical execution of four NEW prediction-matched sampling plans.

Saved own-consume/prepare never runs a model or simulator. No old outcomes,
neural runner, padding, extra replay or scientific-admission authority.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import p1_moving_extended_baseline_physical_20261027_v1 as helper

NAME = 'moving_prediction_matched_sampling_physical_20261027_v1'
AUTH = 'MAIN_PREDICTION_MATCHED_SAMPLING_PHYSICS_ONLY'
AUTHORITY = 'moving_prediction_matched_sampling_cuda_20261027_v1'
ARMS = ['CEM', 'kernel_faithful_mppi_nominal']
kernel, ROOT = helper.kernel, helper.ROOT
require, sha, raw, source = helper.require, helper.sha, helper.raw, helper.source
CAP, FLOOR, FINAL, PREPARE_CAP = 32*1024**2, 200*1024**2, 512*1024, 256*1024
SOURCE_SHA = helper.SOURCE_SHA
HELPER_SHA = 'a0f7348ebe8c6e061b5d63a1ba4a8ecb064a26e7f7a1ef4be57fe8b07270d799'
fields = helper.fields


def hash_literal(value):
    require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
            'external literal SHA256 required, never infer pending authority')
    return value


def absolute(value):
    require(type(value) is str and Path(value).is_absolute() and str(Path(value).resolve()) == value,
            'canonical absolute path, no symlink components')
    return value


def read(path, cap=PREPARE_CAP):
    path = Path(absolute(str(path)))
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= cap, 'bounded regular JSON')
    data = path.read_bytes()
    require(len(data) <= cap, 'read size cap')
    # Producer canonical parser rejects non-finite values when re-encoding.
    import json
    def unique(pairs):
        result = {}
        for k, v in pairs:
            require(k not in result, 'duplicate JSON key'); result[k] = v
        return result
    value = json.loads(data, object_pairs_hook=unique,
                       parse_constant=lambda _: require(False, 'nonfinite JSON'))
    raw(value)
    return value


def merge(bindings, additions):
    for path, h in additions.items():
        absolute(path); hash_literal(h)
        require(path not in bindings or bindings[path] == h, 'nonconflicting identity seals')
        bindings[path] = h


def authority_files(entry):
    """MAIN supplies FINAL producer source/tests/doc seals; no guessed constants."""
    return {str(ROOT/'experiments/spectral_reachability'/(AUTHORITY+'.py')): hash_literal(entry['source_sha256']),
            str(ROOT/'tests'/('test_'+AUTHORITY+'.py')): hash_literal(entry['tests_sha256']),
            str(ROOT/'innerdocs'/(AUTHORITY+'.md')): hash_literal(entry['innerdoc_sha256'])}


def ownconsume(entry):
    fields(entry, 'authority output report_sha256 source_sha256 tests_sha256 innerdoc_sha256')
    require(entry['authority'] == AUTHORITY, 'only NEW matched sampling own consumer')
    frozen = authority_files(entry)
    for path, h in frozen.items(): require(sha(path) == h, 'MAIN final producer seal BEFORE import')
    output = Path(absolute(entry['output'])); expected = hash_literal(entry['report_sha256'])
    require(sha(output/'report.json') == expected, 'MAIN external actual report SHA')
    module = importlib.import_module('experiments.spectral_reachability.'+AUTHORITY)
    path = ROOT/'experiments/spectral_reachability'/(AUTHORITY+'.py')
    require(Path(module.__file__).resolve() == path and sha(path) == entry['source_sha256'], 'whitelisted own module')
    receipt = module.consume(output, expected)  # saved NumPy reconstruction ONLY; no model/physics
    expected_resources = {k: dict(attempted=576, completed=576)
                          for k in ('predict_API', 'predictor', 'action_encoder', 'pred_proj')}
    require(receipt['schema'] == AUTHORITY+'.consume' and receipt['P1_admission'] is False and
            receipt['equal_total_compute'] is False and receipt['official_MPPI'] is False and
            receipt['prediction_calls_and_samples_matched_between_sampling_arms'] is True and
            receipt['resources'] == expected_resources and
            receipt['predictor_samples'] == dict(attempted=576, completed=576), 'complete actual matched sampling receipt')
    rows = deepcopy(receipt['rows'])
    require(isinstance(rows, list) and len(rows) == 4, 'ALL four plans, no old denominator')
    require([(r['source_slot'], r['arm']) for r in rows] == [(s, a) for s in (0, 1) for a in ARMS],
            'exact four ordered root/arm identities')
    for row in rows:
        require(type(row['source_slot']) is int and row['schema'] == AUTHORITY+'.model_plan' and
                row['whole_run_report_sha256'] == expected and row['with_target'] is False and
                row['physical_execution'] is False and row['selection_uses_physics'] is False and
                row['P1_admission'] is False and row['official_MPPI'] is False, 'own plan provenance')
        content = {k: v for k, v in row.items() if k not in ('selection_sha256', 'whole_run_report_sha256')}
        require(source.digest(content) == row['selection_sha256'], 'original own selected plan seal')
        kernel.commands(row['commands'])
    report = read(output/'report.json', CAP)
    prepared = Path(absolute(report['prepared']))
    require(sha(prepared) == report['prepare_sha256'], 'producer actual prepare seal')
    p = read(prepared)
    require(p['schema'] == AUTHORITY+'.prepare' and len(p['slots']) == 2, 'new producer prepare/two slots')
    for row in rows:
        slot = p['slots'][row['source_slot']]
        require(all(row[k] == slot[k] for k in ('source_slot', 'task_id', 'blind_sha256', 'task_export',
                    'task_sha256', 'task_contract_sha256')), 'producer slot/task/history contract join')
    bindings = {}
    merge(bindings, p['bindings_sha256']); merge(bindings, frozen)
    merge(bindings, {str(prepared): report['prepare_sha256'], str(output/'report.json'): expected})
    for name, h in report['artifact_sha256'].items():
        require(Path(name).name == name and name not in ('', '.', '..'), 'flat producer artifact')
        merge(bindings, {str(output/name): h})
    for path, h in bindings.items(): require(sha(path) == h, 'complete saved producer closure')
    return rows, bindings


def task_bridge(row, ref, bindings):
    # Reads the moving task export: exact original history/goal RGB, raw warmup,
    # task-contract SHA, identity192 Q, projected logits -> literal float32 raw10.
    helper.task_bridge(row, ref, bindings)


def selection(m, data):
    fields(m, 'schema sourcepack sourcepack_sha256 source_prepared source_prepare_sha256 roots arms authorities selection_fixed_before_physics physical_outcome_selection wall_seconds')
    require(m['schema'] == NAME+'.selection' and m['sourcepack_sha256'] == SOURCE_SHA and
            m['arms'] == ARMS and m['selection_fixed_before_physics'] is True and
            m['physical_outcome_selection'] is False, 'fixed BOTH new sampling arms; no outcome subset')
    require(type(m['wall_seconds']) is int and 0 < m['wall_seconds'] <= 600, 'wall<=600')
    require(len(data['references']) == 2 and m['roots'] == [dict(root_slot=i, root_sha256=r['root_sha256'])
            for i, r in enumerate(data['references'])] and all(type(r['root_slot']) is int for r in m['roots']),
            'original two known moving roots/order')
    require(isinstance(m['authorities'], list) and len(m['authorities']) == 1, 'one NEW producer only')
    entry = m['authorities'][0]; rows, bindings = ownconsume(entry)
    require([(r['source_slot'], r['arm']) for r in rows] == [(s, a) for s in (0, 1) for a in ARMS]
            and all(type(r['source_slot']) is int for r in rows), 'ALL four distinct new plans')
    plans = []
    for row in rows:
        s = row['source_slot']; ref = data['references'][s]
        task_bridge(row, ref, bindings)
        plan = dict(root_slot=s, root_sha256=ref['root_sha256'], arm=row['arm'],
            commands=kernel.commands(row['commands']).tolist(), task_id=row['task_id'],
            blind_sha256=row['blind_sha256'], task_export=row['task_export'], task_sha256=row['task_sha256'],
            task_contract_sha256=row['task_contract_sha256'], compiler_authority=AUTHORITY,
            compiler_report=str(Path(absolute(entry['output']))/'report.json'),
            compiler_report_sha256=hash_literal(entry['report_sha256']),
            compiler_selection_sha256=row['selection_sha256'], own_plan_sha256=source.digest(row))
        plan['selection_digest'] = source.digest(plan); plans.append(plan)
    return plans, bindings


def contract(wall=600):
    require(type(wall) is int and 0 < wall <= 600, 'wall<=600')
    c = kernel.low.contract(4, wall)
    c['caps'].update(environment_construct=4, environment_close=4)
    c.update(arms=list(ARMS), output_cap_bytes=CAP, disk_floor_bytes=FLOOR, final_reserve_bytes=FINAL,
        prepare_cap_bytes=PREPARE_CAP, warmup_raw=kernel.moving.warmup_commands().tolist(),
        sourcepack_sha256=SOURCE_SHA, padding=0, retries=0, extra_replay=0,
        new_neural_calls=0, training=False, with_target=False, goal_RGB_visible=False,
        constructor_scope='source-sealed inert native constructor; partial constructor work unknown, never zero-filled',
        native_scope='reset setters/goal renders/space.step hooked AFTER pristine constructor; explicit render is a separate overlapping view',
        wall_scope='from Store creation, before dispatch and regular write; cleanup/final report exempt, blocked native calls not preemptible',
        count_scope='actual callback entry/return, not journal reservations; failures never refund',
        goal_binding='evaluator metadata/goal pixels only; no goal_pose insertion, new physics or target drawing',
        selection_scope='NEW four winners only, two KNOWN moving calibration roots x CEM/local nominalMPPI',
        matched_claim='inherited sampling prediction API calls AND predictor forward samples only; each144 total576, batch1',
        inherited_prediction_cost='separate producer own-report closure; AE/pred_proj overlap, never added to API/samples',
        equal_total_compute=False, equal_FLOPs=False, equal_AD_compute=False,
        P1_admission=False, independent_confirmation=False)
    return c


def physical_files():
    return sorted(set(helper.physical_files()+[Path(__file__).resolve()]))


def material(manifest, expected):
    path = Path(absolute(str(manifest))); hash_literal(expected)
    require(path.stat().st_size <= PREPARE_CAP and sha(path) == expected, 'MAIN external bounded manifest SHA')
    require(sha(helper.__file__) == HELPER_SHA, 'sealed physical primitives')
    for f, h in helper.SEALED_HELPERS.items(): require(sha(f) == h, 'sealed lower physical primitives')
    m = read(path)
    require(m['schema'] == NAME+'.selection' and m['sourcepack_sha256'] == SOURCE_SHA, 'only original moving source')
    for k in ('sourcepack', 'source_prepared'): absolute(m[k])
    data = kernel.load_source(m['sourcepack'], m['sourcepack_sha256'], m['source_prepared'], m['source_prepare_sha256'])
    plans, bindings = selection(m, data)
    merge(bindings, data['bindings']); merge(bindings, {str(path): expected})
    native, h = helper.native_constructor_contract(); merge(bindings, {native: h})
    for f in physical_files()+[ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')]:
        merge(bindings, {str(f.resolve()): sha(f)})
    for f, h in bindings.items(): require(sha(f) == h, 'final source/model/task/artifact seals')
    return dict(schema=NAME+'.prepare', manifest=str(path), manifest_sha256=expected, plans=plans,
        bindings=bindings, contract=contract(m['wall_seconds']), sourcepack_sha256=SOURCE_SHA,
        source_prepare_id=source.digest(data['plan']))


def prepare(manifest, expected, output):
    """MAIN only: saved validation and exclusive prepare write, no simulator/model."""
    p = material(manifest, expected); payload = raw(p); path = Path(absolute(str(output)))
    require(len(payload) <= PREPARE_CAP and not path.exists() and not path.is_symlink() and
            shutil.disk_usage(path.parent).free >= FLOOR+len(payload), 'NEW small prepare +200MiB floor')
    with path.open('xb') as f: f.write(payload); f.flush(); os.fsync(f.fileno())
    return dict(prepared=str(path), prepare_sha256=sha(path), bytes=len(payload))


def verify(path, expected):
    path = Path(absolute(str(path))); hash_literal(expected)
    require(sha(path) == expected, 'MAIN external physical prepare SHA'); p = read(path)
    require(p == material(p['manifest'], p['manifest_sha256']) and sha(path) == expected, 'full reconstruction/TOCTOU')
    return p


class Store(helper.Store):
    """Own32MiB/600s bounds, no mutation of inherited Store globals."""
    def __init__(self, output, wall=600):
        contract(wall); self.started = time.monotonic(); self.wall = wall
        self.path = Path(absolute(str(output)))
        require(not self.path.exists() and not self.path.is_symlink() and
                shutil.disk_usage(self.path.parent).free >= FLOOR+CAP, 'NEW32MiB +200MiB floor')
        self.path.mkdir(); self.previous = None; self.index = 0

    def check_time(self):
        require(time.monotonic()-self.started < self.wall, 'wall cap before dispatch/write')

    def used(self):
        files = list(self.path.iterdir())
        require(all(p.is_file() and not p.is_symlink() for p in files), 'flat regular output only')
        return sum(p.stat().st_size for p in files)

    def check(self, size=4096):
        self.check_time()
        require(self.used()+size <= CAP-FINAL and shutil.disk_usage(self.path).free >= FLOOR+size,
                '32MiB cap/final reserve/200MiB floor before dispatch')

    def write(self, name, value, append=False, final=False):
        if not final: self.check_time()
        require(Path(name).name == name and name not in ('', '.', '..') and not (self.path/name).is_symlink(), 'flat output')
        require(self.used()+len(value) <= CAP-(0 if final else FINAL) and
                shutil.disk_usage(self.path).free >= FLOOR+len(value), '32MiB INCLUDING final/partial +200MiB floor')
        with (self.path/name).open('ab' if append else 'xb') as f: f.write(value); f.flush(); os.fsync(f.fileno())
        fd = os.open(self.path, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
        return sha(self.path/name)


class Ops:
    """Actual entries/returns separate from durable predispatch intent, no refunds."""
    def __init__(self, p, store):
        self.p, self.store = p, store
        self.started = getattr(store, 'started', time.monotonic())
        self.where = {}; self.receipts = []; self.fault = False
        self.counts = {k: dict(attempted=0, completed=0) for k in p['contract']['caps']}

    def call(self, kind, fn, **fields):
        require(not self.fault and time.monotonic()-self.started < self.p['contract']['wall_seconds'], 'wall/journal dispatch closed')
        count = self.counts[kind]
        require(count['attempted'] < self.p['contract']['caps'][kind], 'predispatch cap: '+kind)
        self.store.check()
        rec = dict(operation_index=len(self.receipts), resource=kind, **deepcopy(self.where), **deepcopy(fields), status='pending')
        try: self.store.emit('operation_dispatch_intent', receipt=rec)
        except BaseException: self.fault = True; raise
        count['attempted'] += 1; self.receipts.append(rec)
        try: value = fn()
        except BaseException as exc:
            rec.update(status='failed', error_type=type(exc).__name__, message=str(exc))
            try: self.store.emit('operation_failed', receipt=rec)
            except BaseException as secondary: self.fault = True; exc.add_note('failure journal: '+repr(secondary))
            raise
        count['completed'] += 1; rec['status'] = 'returned'
        try: self.store.emit('operation_returned', receipt=rec)
        except BaseException: self.fault = True; raise
        return value


def close_env(env, row, ops, store):
    """Always cleanup after budget/journal failure; still bounded to one close/env."""
    c = ops.counts['environment_close']
    require(c['attempted'] < ops.p['contract']['caps']['environment_close'], 'cleanup cap')
    rec = dict(operation_index=len(ops.receipts), resource='environment_close', **deepcopy(ops.where), status='pending')
    try: store.emit('operation_dispatch_intent', receipt=rec)
    except BaseException as exc: ops.fault = True; kernel.retain(row, exc, 'close_intent_journal')
    c['attempted'] += 1; ops.receipts.append(rec)
    try: env.close()
    except BaseException as exc:
        rec.update(status='failed', error_type=type(exc).__name__, message=str(exc)); kernel.retain(row, exc, 'close')
        try: store.emit('operation_failed', receipt=rec)
        except BaseException as secondary: ops.fault = True; kernel.retain(row, secondary, 'close_failure_journal')
    else:
        c['completed'] += 1; rec['status'] = 'returned'
        try: store.emit('operation_returned', receipt=rec)
        except BaseException as exc: ops.fault = True; kernel.retain(row, exc, 'close_return_journal')


def execute(p, data, store, factory, *, authorization, execution_binding=None):
    require(authorization == AUTH, 'MAIN physics authorization required')
    require(p['contract'] == contract(p['contract']['wall_seconds']) and len(data['references']) == 2 and
            [(x['root_slot'], x['arm']) for x in p['plans']] == [(s, a) for s in (0, 1) for a in ARMS], 'fixed four denominator/contract')
    for plan in p['plans']:
        require(type(plan['root_slot']) is int and plan['root_sha256'] == data['references'][plan['root_slot']]['root_sha256'] and
                plan['selection_digest'] == source.digest({k: v for k, v in plan.items() if k != 'selection_digest'}), 'sealed plan/root join')
        kernel.commands(plan['commands'])
    rows = [dict(branch_index=i, root_slot=x['root_slot'], arm=x['arm'], selected_plan=deepcopy(x),
        selection_digest=x['selection_digest'], status='not_started', first_error=None, secondary_errors=[],
        warmup_returned=0, own_attempted=0, own_returned=0, initial_exact=False, full_horizon=False,
        native_success=None, terminated=None, truncated=None, initial_metrics=None, final_metrics=None,
        final_state=None, endpoint_scope='not_started', PNG=0, checkpoints=[], constructor_internal_work='not_attempted')
        for i, x in enumerate(p['plans'])]
    result = dict(schema=NAME+'.report', status='running', rows=rows, errors=[], contract=p['contract'],
        sourcepack_sha256=p['sourcepack_sha256'], source_prepare_id=p['source_prepare_id'], fixed_branch_denominator=4,
        prepared=None if execution_binding is None else execution_binding['prepared'],
        prepare_sha256=None if execution_binding is None else execution_binding['prepare_sha256'],
        new_neural_calls=0, training=False, new_independent_samples=0, P1_admission=False, P2_admission=False,
        P3_admission=False, scientific_admission=False, independent_confirmation=False, equal_total_compute=False,
        equal_FLOPs=False, compute_advantage=False, walltime_advantage=False)
    ops = Ops(p, store)
    def error(stage, exc): result['errors'].append(dict(stage=stage, type=type(exc).__name__, message=str(exc)))
    try:
        store.emit('run_started', prepare=p, execution_binding=execution_binding)
        for i, (plan, row) in enumerate(zip(p['plans'], rows, strict=True)):
            env = None; holder = []
            ops.where = dict(branch=i, root_slot=plan['root_slot'], arm=plan['arm'], selection_digest=plan['selection_digest'])
            def construct():
                row['constructor_internal_work'] = 'unknown'
                value = factory(); holder.append(value); return value
            try:
                env = ops.call('environment_construct', construct)
                kernel.moving.validate_env(env.unwrapped)
                body = env.unwrapped
                require(all(getattr(body, k, None) is None for k in ('space', 'render_buffer', 'latest_action')), 'pristine constructor BEFORE hooks')
                require(body.render_action is False and body.render_size == 224 and body.render_mode == 'rgb_array', 'target-free RGB224/no marker')
                row.update(status='running', constructor_internal_work='sealed_native_pristine_no_physics')
                with kernel.native_hooks(env, ops, row):
                    kernel.branch(env, data['references'][plan['root_slot']], data['plan'], plan, row, ops, store, i)
            except BaseException as exc: kernel.retain(row, exc, 'setup_or_branch')
            finally:
                if env is None and holder: env = holder[0]
                if env is not None: close_env(env, row, ops, store)
                # Preserve native entry/return even when AFTER-return durable logging fails.
                steps = [r for r in ops.receipts if r.get('branch') == i and r['resource'] == 'public_step']
                row['own_attempted'] = sum(r.get('phase') == 'plan' for r in steps)
                row['own_returned'] = sum(r.get('phase') == 'plan' and r['status'] == 'returned' for r in steps)
                row['warmup_returned'] = sum(r.get('phase') == 'warmup' and r['status'] == 'returned' for r in steps)
                row['full_horizon'] = row['own_returned'] == 10
                if row['status'] == 'failed': row['endpoint_scope'] = 'failed_partial'
                store.emit('branch_closed', receipt=row)
    except BaseException as exc: error('execution', exc)
    try:
        for f, h in p['bindings'].items(): require(sha(f) == h, 'final binding: '+f)
    except BaseException as exc: error('final_bindings', exc)
    clean = not result['errors'] and all(r['status'] in ('horizon_completed', 'native_success', 'native_stopped') for r in rows)
    if clean:
        try:
            c = ops.counts; steps = sum(r['warmup_returned']+r['own_returned'] for r in rows)
            require(all(v['attempted'] == v['completed'] for v in c.values()) and
                c['environment_construct']['completed'] == c['environment_close']['completed'] == 4 and
                c['reset']['completed'] == 8 and c['set_state']['completed'] == 16 and
                c['goal_render']['completed'] == 8 and c['goal_bind']['completed'] == 4 and
                c['public_step']['completed'] == steps and c['physics_step']['completed'] == 16+10*steps and
                c['explicit_render']['completed'] == sum(r['PNG'] for r in rows), 'actual full/earlystop census, never pad partial')
        except BaseException as exc: error('final_census', exc); clean = False
    result.update(status='matched_sampling_physical_recorded_not_admitted' if clean else 'partial_or_failed',
        counts=ops.counts, actual_operation_receipts=ops.receipts,
        full_horizons=sum(r['full_horizon'] for r in rows), native_successes=sum(r['status'] == 'native_success' for r in rows),
        engineering_failures=sum(r['status'] == 'failed' for r in rows), not_started=sum(r['status'] == 'not_started' for r in rows),
        early_stops=sum(r['status'] in ('native_success', 'native_stopped') and not r['full_horizon'] for r in rows),
        wall_seconds=time.monotonic()-ops.started,
        total_internal_physics_scope='unknown_constructor_work' if any(r['constructor_internal_work'] == 'unknown' for r in rows)
            else 'hook_observed_native_work', resource_equality_required=False)
    try:
        store.emit('run_finished', status=result['status'], counts=ops.counts, errors=result['errors'])
        result.update(journal_sha256=sha(store.path/'journal.jsonl'), journal_chain_tail=store.previous)
    except BaseException as exc: error('final_journal', exc); result['status'] = 'partial_or_failed'
    try: store.put('report.json', result, final=True)
    except BaseException as exc: error('final_report', exc); result['status'] = 'partial_or_failed'
    return result


def run(prepared, expected, output, factory, *, authorization):
    require(authorization == AUTH, 'MAIN authorization BEFORE I/O/factory')
    p = verify(prepared, expected); m = read(p['manifest'])
    data = kernel.load_source(m['sourcepack'], m['sourcepack_sha256'], m['source_prepared'], m['source_prepare_sha256'])
    store = Store(output, p['contract']['wall_seconds'])
    binding = dict(prepared=absolute(str(prepared)), prepare_sha256=expected)
    store.put('execution_binding.json', binding)
    return execute(p, data, store, factory, authorization=authorization, execution_binding=binding)


def assemble(report, events, p):
    """Pure saved-data ABI for comparison/RGB; preserves all four/partial rows.

    Missing terminal journal or dispatch-return receipts rejects consumption,
    rather than upgrading an unverifiable partial run. No source simulator.
    """
    require(report['schema'] == NAME+'.report' and report['status'] in
            ('matched_sampling_physical_recorded_not_admitted', 'partial_or_failed') and
            report['contract'] == p['contract'] == contract(p['contract']['wall_seconds']) and
            report['sourcepack_sha256'] == p['sourcepack_sha256'] == SOURCE_SHA and
            report['source_prepare_id'] == p['source_prepare_id'] and report['fixed_branch_denominator'] == 4,
            'own complete four-row physical identity')
    for flag in ('P1_admission', 'P2_admission', 'P3_admission', 'scientific_admission',
                 'independent_confirmation', 'equal_total_compute', 'equal_FLOPs', 'training'):
        require(report[flag] is False, 'no scientific/compute upgrade')
    require(report['new_neural_calls'] == report['new_independent_samples'] == 0, 'physical only known roots')
    require(events and events[0]['kind'] == 'run_started' and events[0]['prepare'] == p and
            events[-1]['kind'] == 'run_finished' and events[-1]['status'] == report['status'] and
            events[-1]['counts'] == report['counts'] and events[-1]['errors'] == report['errors'] and
            events[-1]['event_sha256'] == report['journal_chain_tail'], 'durable physical prepare/terminal binding')
    order = [(s, a) for s in (0, 1) for a in ARMS]
    require([(r['root_slot'], r['arm']) for r in report['rows']] == order and
            [(r['root_slot'], r['arm']) for r in p['plans']] == order, 'ALL four original identities, not successes only')
    counts = {k: dict(attempted=0, completed=0) for k in p['contract']['caps']}
    intents, terminals, stack = {}, {}, []
    for event in events:
        kind = event['kind']
        if kind not in ('operation_dispatch_intent', 'operation_returned', 'operation_failed'): continue
        r = event['receipt']; i = r['operation_index']
        require(type(i) is int and r['resource'] in counts, 'literal known meter resource/index')
        if kind == 'operation_dispatch_intent':
            require(i == len(intents) and r['status'] == 'pending', 'sequential predipatch intents')
            intents[i] = r; stack.append(i)
        else:
            require(i in intents and i not in terminals and stack and stack.pop() == i and
                    r['status'] == ('returned' if kind == 'operation_returned' else 'failed'), 'actual nested terminal receipt')
            original = {k: v for k, v in r.items() if k not in ('error_type', 'message')}
            require(dict(original, status='pending') == intents[i], 'actual terminal matches original dispatch identity')
            terminals[i] = r
    require(not stack and len(terminals) == len(intents) and report['actual_operation_receipts'] ==
            [terminals[i] for i in range(len(terminals))], 'every actual entry/return independently durable')
    for r in report['actual_operation_receipts']:
        c = counts[r['resource']]; c['attempted'] += 1; c['completed'] += r['status'] == 'returned'
    require(counts == report['counts'] and all(v['attempted'] <= p['contract']['caps'][k]
            for k, v in counts.items()), 'actual meter/caps, never expected fullcounts on partial')
    checkpoints = []
    closed = [e['receipt'] for e in events if e['kind'] == 'branch_closed']
    require(closed == [r for r in report['rows'] if r['status'] != 'not_started'], 'durable branch rows/no missing denominator')
    for index, (row, plan) in enumerate(zip(report['rows'], p['plans'], strict=True)):
        require(type(row['root_slot']) is int and row['branch_index'] == index and row['selected_plan'] == plan and
                row['selection_digest'] == plan['selection_digest'] == source.digest({k: v for k, v in plan.items() if k != 'selection_digest'}),
                'exact root/task/raw/goal source-selected plan binding')
        steps = [r for r in report['actual_operation_receipts'] if r['resource'] == 'public_step' and r.get('branch') == index]
        own = [r for r in steps if r['phase'] == 'plan']; warm = [r for r in steps if r['phase'] == 'warmup']
        require(row['own_attempted'] == len(own) and row['own_returned'] == sum(r['status'] == 'returned' for r in own) and
                row['warmup_returned'] == sum(r['status'] == 'returned' for r in warm) and
                row['full_horizon'] == (row['own_returned'] == 10), 'row actualattempt-return, not durable intent counts')
        for phase, opset, commands in (('warmup', warm, p['contract']['warmup_raw']), ('plan', own, plan['commands'])):
            require([r['step'] for r in opset] == list(range(1, len(opset)+1)) and len(opset) <= 10 and
                    all(r['raw_action'] == commands[j] for j, r in enumerate(opset)), 'exact contiguous literal raw10 no replay/padding')
        cpsteps = [c['step'] for c in row['checkpoints']]
        require(cpsteps == sorted(set(cpsteps)) and all(s in (0, 5, 10) for s in cpsteps), 'only actual checkpoint0/5/10')
        observations = [e['receipt'] for e in events if e['kind'] == 'fresh_observation' and e.get('branch') == index]
        transitions = [e for e in events if e['kind'] == 'returned_transition' and e.get('branch') == index and e['phase'] == 'plan']
        for c in row['checkpoints']:
            state = source.vector(c['state'])  # finite actual7, never truncated object pose
            require(c['phase'] == 'plan' and c in observations and np.array_equal(state, c['observation_state']) and
                    c['step'] <= row['own_returned'], 'checkpoint actual7 state and returned observation')
            guards = [e for e in events if e['kind'] == 'guard_receipt' and e.get('branch') == index and e['phase'] == 'plan' and e['step'] == c['step']]
            require(len(guards) == 1 and guards[0]['accepted'] is True, 'actual checkpoint guard')
            if c['step'] == 0: require(c['state'] == row['root_state'], 'actual checkpoint0 root')
            else:
                transition = [e for e in transitions if e['step'] == c['step']]
                require(len(transition) == 1 and transition[0]['state'] == c['state'], 'checkpoint bound to actual returned raw command')
            checkpoints.append(dict(branch_index=index, root_slot=row['root_slot'], arm=row['arm'],
                selection_digest=row['selection_digest'], selected_plan=deepcopy(plan), checkpoint=deepcopy(c)))
        if row['initial_exact']:
            require(source.root_hash(row['root_state'], row['goal']) == plan['root_sha256'], 'actual root+goal SHA')
            attached = [e for e in events if e['kind'] == 'goal_attached' and e.get('branch') == index]
            require(len(attached) == 1 and attached[0]['goal'] == row['goal'], 'actual offline goal identity')
            require(all(e['returned_goal_state'] == row['goal'] and e['returned_goal_rgb_sha256'] == attached[0]['goal_rgb_sha256']
                        for e in transitions), 'goal remains fixed in actual native returns')
        if row['final_state'] is not None:
            require(row['final_metrics'] == kernel.metric(row['goal'], source.vector(row['final_state'])), 'original native7 labels, no author-reported fill')
    return checkpoints


def consume(output, expected):
    """Saved-only downstream ABI; MAIN supplies the actual external physical SHA."""
    output = Path(absolute(str(output))); hash_literal(expected)
    files = list(output.iterdir())
    require(all(f.is_file() and not f.is_symlink() for f in files) and sum(f.stat().st_size for f in files) <= CAP,
            'bounded regular physical artifacts')
    require(sha(output/'report.json') == expected, 'external actual physical report SHA')
    report = read(output/'report.json', CAP)
    p = verify(report['prepared'], report['prepare_sha256'])
    binding = read(output/'execution_binding.json')
    require(binding == dict(prepared=report['prepared'], prepare_sha256=report['prepare_sha256']), 'actual physical execution binding')
    journal = output/'journal.jsonl'
    require(sha(journal) == report['journal_sha256'] and journal.stat().st_size <= CAP, 'physical journal SHA/size')
    data = journal.read_bytes(); require(len(data) <= CAP and data.endswith(b'\n'), 'complete bounded journal')
    import json
    events = [json.loads(line) for line in data.splitlines()]; previous = None
    for i, event in enumerate(events):
        require(event['index'] == i and event['previous_sha256'] == previous and event['event_sha256'] ==
                source.digest({k: v for k, v in event.items() if k != 'event_sha256'}), 'physical event hashchain')
        previous = event['event_sha256']
    require(previous == report['journal_chain_tail'], 'physical final chain tail')
    checkpoints = assemble(report, events, p)
    # Validate ALL saved actual render files, not only the downstream5/10 subset.
    from PIL import Image
    for event in events:
        if event['kind'] != 'fresh_observation': continue
        c = event['receipt']; name = c['png']
        require(Path(name).name == name and name not in ('', '.', '..') and sha(output/name) == c['file_sha256'], 'actual PNG bytes seal')
        with Image.open(output/name) as image:
            require(image.size == (224, 224) and image.mode == 'RGB', 'native RGB224 stored image')
            frame = np.asarray(image)
        require(frame.dtype == np.uint8 and kernel.moving.image_sha(frame) == c['pixel_sha256'], 'actual PNG pixel seal')
    bindings = {}; merge(bindings, p['bindings'])
    merge(bindings, {report['prepared']: report['prepare_sha256']})
    hashes = {str(f): sha(f) for f in files}; require(hashes[str(output/'report.json')] == expected, 'report final TOCTOU')
    merge(bindings, hashes)
    for path, h in bindings.items(): require(sha(path) == h, 'final saved physical closure')
    for c in checkpoints: c['png_path'] = str(output/c['checkpoint']['png'])
    return dict(schema=NAME+'.consume', status=report['status'], rows=deepcopy(report['rows']),
        checkpoints=checkpoints, counts=deepcopy(report['counts']), report=report, prepared=p,
        bindings_sha256=bindings, physical_report_sha256=expected, fixed_branch_denominator=4,
        checkpoint_complete=all([c['step'] for c in r['checkpoints']] == [0, 5, 10] for r in report['rows']),
        scientific_admission=False, P1_admission=False, independent_confirmation=False,
        equal_total_compute=False, equal_FLOPs=False, new_neural_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'run', 'consume'))
    for k in ('manifest', 'manifest-sha', 'prepared', 'prepare-sha', 'output', 'report-sha'): parser.add_argument('--'+k)
    parser.add_argument('--authorize-physics', action='store_true'); a = parser.parse_args()
    if a.command == 'prepare': r = prepare(a.manifest, a.manifest_sha, a.output)
    elif a.command == 'verify': r = dict(schema=verify(a.prepared, a.prepare_sha)['schema'])
    elif a.command == 'consume': r = consume(a.output, a.report_sha)
    else:
        require(a.authorize_physics, 'MAIN CLI physics authorization')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401 -- registration only, no model loading
            return gym.make('swm/PushT-v1', render_mode='rgb_array', resolution=224, with_target=False, render_action=False)
        print('Four NEW matched-sampling winners, known moving2 physical calibration only; no FLOPs/P1 claim.', flush=True)
        r = run(a.prepared, a.prepare_sha, a.output, factory, authorization=AUTH)
    print(source.canonical({k: v for k, v in r.items() if k not in ('rows', 'actual_operation_receipts')}))
    if a.command == 'run' and r['status'] != 'matched_sampling_physical_recorded_not_admitted': raise SystemExit(2)


if __name__ == '__main__': main()
