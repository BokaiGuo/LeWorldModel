"""MAIN-only CUDA search plus independent five-step winner-prefix validation.

Prepare/verify/consume are saved arithmetic only. No training, physics, child
signals, outcome tuning, or scientific release. Calls the sealed panel directly.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
import os
from pathlib import Path
import time

import numpy as np

from experiments.spectral_reachability import agent_p3_fixed_execution_contract_20261028_v1 as fixed

common, native, signal, core = fixed.common, fixed.native, fixed.signal, fixed.core
require, canonical, digest, sha = fixed.require, fixed.canonical, fixed.digest, fixed.sha
ROOT, FLAGS = fixed.ROOT, dict(fixed.FLAGS)
NAME = 'agent_p3_fixed_execution_cuda_20261028_v1'
PANEL_SHA = '7cb1cb3a89895755d2bb92cc9dd62de617e4a48471aa388f6fff4eab48a3aae9'
AUTH = 'MAIN_APPROVED_HOST_CUDA'
PREFIX = 'selected_prefix_verification'
ARRAY_KINDS = {'io_input', 'io_return', 'program_packet', 'program_boundary'}


def own_files():
    return [Path(__file__).resolve(), ROOT/f'tests/{NAME}_test.py', ROOT/f'innerdocs/{NAME}.md']


def material(contract_prepared, expected):
    require(sha(fixed.__file__) == PANEL_SHA, 'unchanged frozen panel')
    panel = fixed.verify(contract_prepared, expected)
    files = set(map(Path, panel['sources'])) | set(own_files()) | {Path(contract_prepared).resolve()}
    return dict(schema=NAME+'.prepare', contract_prepared=str(Path(contract_prepared).resolve()),
        contract_prepare_sha256=expected, panel=panel,
        sources={str(f): sha(f) for f in sorted(files)},
        profile=panel['profile'], authorization=AUTH,
        scope='root-only model search and five-step prefix; not physical or full C3',
        saved_verification=dict(run_upstream_array_SVD_calls=16, consume_upstream_array_SVD_calls=8,
            consume_generator_array_SVD_calls=4, native_new_array_SVD_calls=4,
            semantics='saved verification separate from native ledger; no neural CPU work'),
        missing_dependencies=['MAIN source review and explicit HOST CUDA execution with strict saved consume',
            'new fixed-five-step metered physical consumer with exact recipe/root/goal bindings',
            'independent P1 and applicable P2 admission; certified fresh cohort',
            'new child observations and paid child-local J adapters for any full C3 tree'], **FLAGS)


def exclusive(path):
    p = Path(path).absolute()
    require(p.parent == ROOT/'results' and p.name.startswith(NAME) and not p.exists() and not p.is_symlink(), 'exclusive new own result')
    return p


def write_new(path, value):
    p = exclusive(path); raw = canonical(value)
    require(len(raw) <= 16*1024**2, 'bounded prepare')
    with p.open('xb') as f:
        require(f.write(raw) == len(raw), 'complete prepare write'); f.flush(); os.fsync(f.fileno())
    signal.fsync_dir(p.parent)
    return dict(path=str(p), sha256=sha(p))


def verify(path, expected):
    core.hash_value(expected); require(sha(path) == expected, 'external actual prepare SHA')
    p = signal.read(path)
    require(canonical(p) == canonical(material(p['contract_prepared'], p['contract_prepare_sha256'])) and sha(path) == expected,
        'complete frozen panel/source/weight/input/budget rebuild')
    return p


def binding(prepared, expected, p):
    return dict(schema=NAME+'.binding', prepared=str(Path(prepared).resolve()), prepare_sha256=expected,
        source_manifest_sha256=digest(p['sources']), panel_sha256=PANEL_SHA,
        queue_sha256=digest(p['panel']['queue']), authorization=AUTH)


def weight_hashes(p):
    policy = Path(p['panel']['base']['policy']).resolve()
    return {str(policy/n): sha(policy/n) for n in ('weights.pt', 'config.json')}


def initial_result(panel, rows):
    return dict(status='failed_or_incomplete', rows=rows, winners=[],
        prefixes=[dict(slot=s, mode=mode, status='not_started', request=None, error=None) for s in (0, 1) for mode in common.MODES],
        resource_receipt=None, equal_requested_execution_horizon=True,
        equal_actual_physical_execution_horizon=None, actual_physics_status='unknown_not_run', physical_result=None,
        error=None, **FLAGS)


def counts(result):
    return dict(task_methods=4, candidates=len(result['rows']),
        candidate_returns=sum(r['status'] == 'model_replayed_not_admitted' for r in result['rows']),
        selected_prefixes=4, prefix_returns=sum(p['status'] == 'model_verified_physical_blocked' for p in result['prefixes']))


def load_backend(p, store, authorization):
    """Called ONLY after explicit MAIN authorization and durable census/binding."""
    require(authorization == AUTH, 'MAIN explicit HOST CUDA loader authorization required')
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    host = require_cuda_allocation('cuda:0'); device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    store.memory = lambda: (torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device))
    store.check()
    model = load_compatible_pretrained(p['panel']['base']['policy'], device=device).eval().requires_grad_(False)
    # Return ownership immediately; every subsequent failure is finalized by run.
    return model, device, host, torch, sdpa_kernel, SDPBackend


def run(prepared, expected, output, authorization):
    require(authorization == AUTH, 'MAIN review and explicit HOST CUDA authorization required')
    started = time.monotonic(); p = verify(prepared, expected); output = exclusive(output)
    panel = p['panel']; base = panel['base']; store = signal.Store(output, p['profile']); store.started = started
    ledger = core.Ledger(p['profile']['resource_caps'], sink=store.event)
    rows = native.census(base['jobs']); result = initial_result(panel, rows)
    errors = []; model = None; adapters = []; before = after = host = weights_before = weights_after = None
    resources = None; sources_ok = False
    try:
        store.put('binding.json', binding(prepared, expected, p))
        store.event(dict(kind='denominator', rows=deepcopy(rows), prefixes=deepcopy(result['prefixes'])))
        store.event(dict(kind='sources_before', sha256=digest(p['sources'])))
        weights_before = weight_hashes(p)
        require(all(p['sources'][f] == h for f, h in weights_before.items()), 'pre-run sealed weights/config')
        store.event(dict(kind='weights_before', hashes=weights_before))
        model, device, host, torch, sdpa_kernel, SDPBackend = load_backend(p, store, authorization)
        before = signal.model_hash(model)
        require(before['original_state_dict'] == base['expected_model_hash'], 'original frozen model identity')
        store.event(dict(kind='model_before', hashes=before))
        for module in model.modules():
            require(str(Path(inspect.getfile(type(module))).resolve()) in p['sources'], 'presealed native class source closure')
        adapters = [native.NativeProgram(model, device, task, store) for task in base['tasks']]
        with torch.autocast(device_type='cuda', enabled=False), sdpa_kernel(SDPBackend.MATH):
            result = fixed.execute_panel(panel, lambda s: adapters[s], ledger, store, rows)
        torch.cuda.synchronize(device); store.check()
        require(result['status'] == 'fixed_execution_model_ready_physical_blocked', 'panel failure: '+str(result['error']))
        resources = fixed.audit_budget(panel, ledger.receipt())
    except BaseException as exc:
        errors.append(dict(stage='execution', message=type(exc).__name__+': '+str(exc)))
    finally:
        for row in rows:
            if row['status'] == 'not_started': row['status'] = 'blocked_not_dispatched'
            elif row['status'] == 'attempted': row['status'] = 'failed'
        for prefix in result['prefixes']:
            if prefix['status'] == 'not_started': prefix['status'] = 'blocked_not_dispatched'
            elif prefix['status'] == 'attempted': prefix.update(status='failed', error=str(errors))
        if model is not None:
            try:
                for adapter in adapters: adapter.frozen()
                after = signal.model_hash(model)
                require(before == after, 'unchanged frozen state/all persistent and nonpersistent buffers')
                store.event(dict(kind='model_after', hashes=after))
            except BaseException as exc: errors.append(dict(stage='model_final', message=str(exc)))
        try:
            weights_after = weight_hashes(p)
            require(weights_after == weights_before, 'unchanged weights/config before-after')
            store.event(dict(kind='weights_after', hashes=weights_after))
        except BaseException as exc: errors.append(dict(stage='weights_final', message=str(exc)))
        try:
            verify(prepared, expected); store.check(); sources_ok = True
            store.event(dict(kind='sources_final', sha256=digest(p['sources'])))
        except BaseException as exc: errors.append(dict(stage='sources_or_limits_final', message=str(exc)))
        try: store.close(); store.check()
        except BaseException as exc: errors.append(dict(stage='close_or_final_limits', message=str(exc)))
        complete = not errors and sources_ok and resources is not None and result['status'] == 'fixed_execution_model_ready_physical_blocked'
        result['resource_receipt'] = ledger.receipt()
        report = dict(result, schema=NAME+'.report', status='saved_model_prefix_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).resolve()), prepare_sha256=expected, errors=errors,
            counts=counts(result), per_branch=resources, model_before=before, model_after=after,
            weights_before=weights_before, weights_after=weights_after,
            sources_verified=sources_ok, source_manifest_sha256=digest(p['sources']), host=host,
            source_manifest_before=digest(p['sources']), source_manifest_after=digest(p['sources']) if sources_ok else None,
            fixed_H1_cost_minimal_baseline=False, same_closed_loop_success_claim=False,
            equal_planning_horizon=common.equal_execution_horizon(base['branches']),
            saved_verification=p['saved_verification'],
            wall_seconds=store.clock()-store.started, CUDA_peak_bytes=store.memory() if store.memory else None,
            journal_tail=store.tail, artifact_sha256={n: sha(output/n) for n in ('binding.json', 'events.jsonl')})
        store.put('report.json', report)
    require(complete, 'partial/failed saved with full denominator; no retry: '+repr(errors))
    return dict(output=str(output), report_sha256=sha(output/'report.json'))


def search_receipt(receipt, caps):
    """Read-only verification projection; NOT a second run or a new receipt."""
    result = deepcopy(receipt); result['caps'] = caps
    ops = [e for e in receipt['events'] if e['scope'].get('operation') != PREFIX]
    result['events'] = ops
    result['attempted'] = result['completed'] = {a: sum(e['delta'][a] for e in ops) for a in core.AXES}
    result['inclusive_operation_wall_ns'] = {r: sum(e['ended_ns']-e['started_ns'] for e in ops if e['resource'] == r)
        for r in sorted({e['resource'] for e in ops})}
    core.audit_ledger(result)
    return result


def validate_prefix_arrays(task, winner, path, local):
    """One genuinely paid whole packet, never invent a segmented replay."""
    require(all(e['phase'] == 'whole' and type(e['step']) is int and e['step'] == 0 for e in local), 'one prefix packet phase/step')
    packets = [e for e in local if e['kind'] == 'program_packet']
    boundaries = [e for e in local if e['kind'] == 'program_boundary']
    require(len(packets) == len(boundaries) == 1 and len(local) == 10, 'one packet/boundary and four native IO pairs')
    packet = {k: signal.unblob(v) for k, v in packets[0]['arrays'].items()}
    end = {k: signal.unblob(v) for k, v in boundaries[0]['arrays'].items()}
    require(set(packet) == {'latent', 'history', 'raw', 'mean', 'std', 'token', 'outgoing'} and set(end) == {'latent', 'history'}, 'exact prefix buffer names')
    require(packets[0]['operation_index'] is boundaries[0]['operation_index'] is None, 'packet not substituted for module operation')
    raw = np.asarray(winner['actions'][:5], np.float32)
    mean, std = (np.asarray(task[k], np.float32) for k in ('action_mean', 'action_std'))
    latent = np.asarray(task['root_context']['latent_history'], np.float32)[None]
    history = np.asarray(task['root_context']['action_history'], np.float32)[None]
    token = ((raw-mean)/std).reshape(10); outgoing = np.concatenate((history, token[None, None]), axis=1)
    for k, value in dict(raw=raw, mean=mean, std=std, latent=latent, history=history, token=token, outgoing=outgoing).items():
        require(np.array_equal(packet[k], value), 'exact prefix raw/normalizer/root buffer: '+k)
    def get(resource, returned=False):
        group = [e for e in local if e['kind'] == ('io_return' if returned else 'io_input') and any(k.startswith(resource+':') for k in e['arrays'])]
        require(len(group) == 1, 'one real prefix module IO')
        return {k: signal.unblob(v) for k, v in group[0]['arrays'].items()}
    ai, ao = get('action_encoder'), get('action_encoder', True)
    pi, po = get('predict'), get('predict', True)
    ni, no = get('predictor'), get('predictor', True)
    qi, qo = get('pred_proj'), get('pred_proj', True)
    for group in (packet, end, ai, ao, pi, po, ni, no, qi, qo):
        require(all(a.dtype == np.float32 for a in group.values()), 'original prefix float32 precision')
    require(set(ai) == {'action_encoder:0'} and set(ao) == {'action_encoder:output'} and
        set(pi) == {'predict:latent', 'predict:act_emb'} and set(po) == {'predict:output'} and
        set(ni) == {'predictor:0', 'predictor:1'} and set(no) == {'predictor:output'} and
        set(qi) == {'pred_proj:0'} and set(qo) == {'pred_proj:output'}, 'exact prefix module ABI')
    require(np.array_equal(ai['action_encoder:0'], outgoing) and np.array_equal(pi['predict:latent'], latent) and
        np.array_equal(pi['predict:act_emb'], ao['action_encoder:output']) and np.array_equal(ni['predictor:0'], latent) and
        np.array_equal(ni['predictor:1'], ao['action_encoder:output']) and
        np.array_equal(qi['pred_proj:0'], no['predictor:output'].reshape(3, 192)) and
        np.array_equal(qo['pred_proj:output'].reshape(1, 3, 192), po['predict:output']), 'prefix AE/predict/predictor/projection closure')
    require(np.array_equal(end['latent'], np.concatenate((latent[:, 1:], po['predict:output'][:, -1:]), axis=1)) and
        np.array_equal(end['history'], outgoing[:, 1:]) and np.array_equal(end['latent'][0], path[0]['latent_history']) and
        np.array_equal(end['history'][0], path[0]['action_history']), 'exact prefix endpoint native closure')


def validate_saved(p, r, events):
    panel, base = p['panel'], p['panel']['base']; receipt = r['resource_receipt']
    require(panel['contract'] == fixed.contract() and panel['queue'] == fixed.queue(base) and panel['profile'] == fixed.profile(base['profile']), 'frozen execution panel')
    require(r['equal_requested_execution_horizon'] is True and r['equal_actual_physical_execution_horizon'] is None and
        r['actual_physics_status'] == 'unknown_not_run' and r['physical_result'] is None and
        r['equal_planning_horizon'] is common.equal_execution_horizon(base['branches']) and
        r['fixed_H1_cost_minimal_baseline'] is r['same_closed_loop_success_claim'] is False and r['error'] is None,
        'requested five steps only, no physical or cheap fixed baseline claims')
    require(r['per_branch'] == fixed.audit_budget(panel, receipt), 'all paid signal/search/prefix axes')
    core.keys(receipt, ('caps', 'attempted', 'completed', 'events', 'wall_ns', 'backward_engine_calls', 'cuda_ns',
        'FLOPs', 'inclusive_operation_wall_ns', 'wall_semantics', 'physics_steps', 'unknowns_are_not_zero'))
    row_keys = ('slot', 'mode', 'index', 'horizon', 'status', 'actions', 'whole', 'segmented', 'cost', 'error')
    for row in r['rows']: core.keys(row, row_keys)
    require(r['counts'] == counts(r) and r['counts'] == dict(task_methods=4, candidates=len(base['jobs']),
        candidate_returns=len(base['jobs']), selected_prefixes=4, prefix_returns=4), 'full successful four-branch denominator')
    initial = initial_result(panel, native.census(base['jobs']))
    require(events[0] == dict(kind='denominator', rows=initial['rows'], prefixes=initial['prefixes']) and
        len([e for e in events if e['kind'] == 'denominator']) == 1 and
        [e for e in events if e['kind'] == 'fixed_execution_denominator'] == [dict(kind='fixed_execution_denominator', queue=panel['queue'])], 'durable predispatch full candidate and prefix census')
    allowed = {'denominator', 'fixed_execution_denominator', 'bank', 'candidate_request', 'candidate_return',
        'model_before', 'model_after', 'weights_before', 'weights_after', 'sources_before', 'sources_final', 'resource_attempt', 'resource_return',
        'selected_prefix_attempt', 'selected_prefix_return'} | ARRAY_KINDS
    require(all(e['kind'] in allowed for e in events), 'unknown/failure/outcome event forbidden')
    # Enforce all array scopes before search/prefix verification; no orphan buffer.
    search_scopes = [{k: j[k] for k in ('slot', 'mode', 'index', 'horizon')} for j in base['jobs']]
    prefix_scopes = [dict(slot=w['slot'], mode=w['mode'], index=w['index'], horizon=1, operation=PREFIX) for w in r['winners']]
    for e in events:
        if e['kind'] in ARRAY_KINDS:
            core.keys(e, ('kind', 'scope', 'phase', 'step', 'operation_index', 'arrays'))
            require(e['scope'] in search_scopes+prefix_scopes, 'only actual search/prefix array scope')
            require(e['phase'] in ('whole', 'segmented') and type(e['step']) is int and
                0 <= e['step'] < e['scope']['horizon'], 'only actual packet phases/steps')
    # Original operation order, not merely a multiset of plausible IO arrays.
    stack = []; operations = receipt['events']; attempts = []; active_prefix = None
    for op in operations:
        core.keys(op, ('index', 'scope', 'resource', 'shape', 'directions', 'delta', 'status', 'started_ns', 'ended_ns', 'error'))
        require(op['error'] is None and op['scope'] in search_scopes+prefix_scopes, 'only original returned operation scopes')
    for e in events:
        if e['kind'] == 'selected_prefix_attempt':
            require(not stack and active_prefix is None, 'prefix attempt before real dispatch')
            active_prefix = (e['slot'], e['mode'])
        elif e['kind'] == 'selected_prefix_return':
            require(not stack and active_prefix == (e['prefix']['slot'], e['prefix']['mode']), 'prefix returned only after its paid native calls')
            active_prefix = None
        elif e['kind'] == 'resource_attempt':
            op = e['entry']; attempts.append(op['index']); stack.append(op['index'])
            if op['scope'].get('operation') == PREFIX:
                require(active_prefix == (op['scope']['slot'], op['scope']['mode']), 'prefix call inside actual attempt/return envelope')
        elif e['kind'] == 'resource_return':
            require(stack and stack.pop() == e['entry']['index'], 'native nested return order')
        elif e['kind'] in ARRAY_KINDS:
            require(stack, 'array must be within real paid operation')
            op = operations[stack[-1]]
            require(op['scope'] == e['scope'], 'array within same operation scope')
            if e['kind'].startswith('program_'):
                require(op['resource'] == 'replay' and e['operation_index'] is None, 'packet inside actual replay only')
            else:
                require(e['operation_index'] == op['index'] and op['resource'] in ('action_encoder', 'predict', 'predictor', 'pred_proj'), 'IO at actual native operation entry/return')
                keys = {op['resource']+':output'} if e['kind'] == 'io_return' else {
                    'action_encoder': {'action_encoder:0'}, 'predict': {'predict:latent', 'predict:act_emb'},
                    'predictor': {'predictor:0', 'predictor:1'}, 'pred_proj': {'pred_proj:0'}}[op['resource']]
                require(set(e['arrays']) == keys, 'exact native module ABI buffers')
    require(not stack and active_prefix is None and attempts == list(range(len(operations))), 'no dangling/reordered native operations')
    prefix_start = next((i for i, e in enumerate(events) if e['kind'] == 'selected_prefix_attempt'), len(events))
    require(not any(e['kind'] in ('bank', 'candidate_request', 'candidate_return') for e in events[prefix_start:]), 'all search before winner-prefix calls')
    search_events = [e for e in events if e['kind'] not in {'fixed_execution_denominator', 'weights_before', 'weights_after', 'sources_before',
        'selected_prefix_attempt', 'selected_prefix_return'} and
        e.get('scope', e.get('entry', {}).get('scope', {})).get('operation') != PREFIX]
    search_events[0] = dict(kind='denominator', rows=native.census(base['jobs']))
    sr = search_receipt(receipt, base['profile']['resource_caps'])
    winners = common.winner_rows(base, r['rows'])
    augmented = [dict(w, task_binding_sha256=digest(base['tasks'][w['slot']]['binding']),
        root_context_sha256=digest(base['tasks'][w['slot']]['root_context'])) for w in winners]
    require(r['winners'] == augmented, 'unchanged stable model-only winners and exact task identity')
    view = dict(rows=r['rows'], winners=winners, resource_receipt=sr, per_branch=common.accounting(base, sr, r['rows']),
        equal_execution_horizon=common.equal_execution_horizon(base['branches']), fixed_H1_cost_minimal_baseline=False,
        same_closed_loop_success_claim=False, comparison_scope=common.rule()['comparison_scope'])
    common.validate_saved(base, view, search_events)
    # Full ledger/IO bijection includes the REAL prefix operations (not projection).
    native.validate_arrays(base, [], events, receipt)
    expected_attempts = []; expected_returns = []
    require([(x['slot'], x['mode']) for x in r['prefixes']] == [(s, m) for s in (0, 1) for m in common.MODES], 'four ordered prefix slots')
    for prefix, winner, scope in zip(r['prefixes'], augmented, prefix_scopes, strict=True):
        core.keys(prefix, ('slot', 'mode', 'status', 'request', 'error'))
        require(prefix['status'] == 'model_verified_physical_blocked' and prefix['error'] is None, 'actual prefix returned')
        row = next(x for x in r['rows'] if all(x[k] == winner[k] for k in ('slot', 'mode', 'index')))
        path = row['whole'][:1]; task = base['tasks'][winner['slot']]
        request = fixed.execution_request(task, dict(winner, prefix_context_sha256=digest(path[0])), path)
        require(prefix['request'] == request, 'exact selected raw actions/full plan/first endpoint/request, no replacement')
        expected_attempts.append(dict(kind='selected_prefix_attempt', slot=winner['slot'], mode=winner['mode'], commands_sha256=digest(winner['actions'][:5])))
        expected_returns.append(dict(kind='selected_prefix_return', prefix=prefix, path=path))
        local = [e for e in events if e['kind'] in ARRAY_KINDS and e['scope'] == scope]
        validate_prefix_arrays(task, winner, path, local)
        ops = [e for e in receipt['events'] if e['scope'] == scope]
        require([e['resource'] for e in ops] == ['replay', 'action_encoder', 'predict', 'predictor', 'pred_proj', 'metric'], 'one actually paid prefix replay and metric, no padded/double replay')
    require([e for e in events if e['kind'] == 'selected_prefix_attempt'] == expected_attempts and
        [e for e in events if e['kind'] == 'selected_prefix_return'] == expected_returns, 'all exact durable prefix attempts/results')


def consume(output, expected):
    output = Path(output).absolute(); core.hash_value(expected)
    require(sha(output/'report.json') == expected, 'external actual report SHA')
    r = signal.read(output/'report.json'); p = verify(r['prepared'], r['prepare_sha256'])
    core.keys(r, ('schema', 'status', 'rows', 'winners', 'prefixes', 'resource_receipt', 'error', *FLAGS,
        'equal_requested_execution_horizon', 'equal_actual_physical_execution_horizon', 'actual_physics_status', 'physical_result',
        'prepared', 'prepare_sha256', 'errors', 'counts', 'per_branch', 'model_before', 'model_after', 'weights_before', 'weights_after',
        'sources_verified', 'source_manifest_sha256', 'host', 'fixed_H1_cost_minimal_baseline', 'same_closed_loop_success_claim',
        'source_manifest_before', 'source_manifest_after',
        'equal_planning_horizon', 'saved_verification', 'wall_seconds', 'CUDA_peak_bytes', 'journal_tail', 'artifact_sha256'))
    require(r['schema'] == NAME+'.report' and r['status'] == 'saved_model_prefix_not_admitted' and r['errors'] == [] and
        all(r[k] is False for k in FLAGS), 'complete engineering only, no admission')
    require(r['sources_verified'] is True and r['source_manifest_sha256'] == digest(p['sources']) and
        r['source_manifest_before'] == r['source_manifest_after'] == digest(p['sources']) and
        r['saved_verification'] == p['saved_verification'], 'exact source and saved-work accounting')
    for kind in ('model_before', 'model_after'):
        core.keys(r[kind], ('original_state_dict', 'all_buffers'))
        for value in r[kind].values(): core.hash_value(value)
    require(r['model_before'] == r['model_after'] and r['model_before']['original_state_dict'] == p['panel']['base']['expected_model_hash'], 'unchanged original model/state/all buffers')
    require(r['weights_before'] == r['weights_after'] == weight_hashes(p) and
        all(p['sources'][f] == h for f, h in r['weights_before'].items()), 'actual sealed weights/config pre/post')
    names = {'report.json', 'binding.json', 'events.jsonl'}
    require({f.name for f in output.iterdir()} == names and all(not (output/n).is_symlink() and (output/n).is_file() for n in names) and
        set(r['artifact_sha256']) == names-{'report.json'}, 'exclusive exact regular artifact set')
    require(signal.read(output/'binding.json') == binding(r['prepared'], r['prepare_sha256'], p), 'complete execution binding')
    for n, h in r['artifact_sha256'].items(): require(sha(output/n) == h, 'saved artifact SHA')
    events, tail = signal.journal(output/'events.jsonl'); require(tail == r['journal_tail'], 'journalchain/report binding')
    for kind in ('model_before', 'model_after', 'weights_before', 'weights_after'):
        require([e for e in events if e['kind'] == kind] == [dict(kind=kind, hashes=r[kind])], 'actual before/after hashes journal')
    require([e for e in events if e['kind'] == 'sources_final'] == [dict(kind='sources_final', sha256=digest(p['sources']))], 'source final journal')
    require([e for e in events if e['kind'] == 'sources_before'] == [dict(kind='sources_before', sha256=digest(p['sources']))], 'source pre-run journal')
    require(type(r['host']) is dict and r['host'].get('device') == 'cuda:0' and r['host'].get('allocation_sum') == 4.0, 'actual HOST CUDA allocation receipt')
    require(type(r['wall_seconds']) in (int, float) and np.isfinite(r['wall_seconds']) and 0 <= r['wall_seconds'] <= p['profile']['wall_seconds'], 'actual wall cap')
    require(type(r['CUDA_peak_bytes']) is list and len(r['CUDA_peak_bytes']) == 2 and all(type(v) is int and 0 <= v <= p['profile'][k]
        for v, k in zip(r['CUDA_peak_bytes'], ('cuda_allocated_bytes', 'cuda_reserved_bytes'), strict=True)), 'actual CUDA memory caps')
    require(sum((output/n).stat().st_size for n in names) <= p['profile']['output_bytes'], 'actual output cap')
    validate_saved(p, r, events)
    require(sha(output/'report.json') == expected and all(sha(f) == h for f, h in p['sources'].items()) and
        all(sha(output/n) == h for n, h in r['artifact_sha256'].items()), 'unchanged final full source/artifact closure')
    return dict(schema=NAME+'.consume', status='saved_prefix_verified_not_admitted', report_sha256=expected,
        prepared=r['prepared'], prepare_sha256=r['prepare_sha256'], source_manifest_sha256=digest(p['sources']),
        root_contexts=[dict(slot=s, binding=deepcopy(t['binding']), context=deepcopy(t['root_context']))
            for s, t in enumerate(p['panel']['base']['tasks'])],
        per_branch=r['per_branch'], counts=r['counts'], winners=r['winners'], prefixes=r['prefixes'],
        equal_requested_execution_horizon=True, equal_actual_physical_execution_horizon=None,
        equal_planning_horizon=r['equal_planning_horizon'], actual_physics_status='unknown_not_run',
        fixed_H1_cost_minimal_baseline=False, same_closed_loop_success_claim=False,
        saved_verification_array_SVD_calls=12, complete_C3_tree=False,
        scientific_effect=None, physical_benefit=None, **FLAGS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--contract-prepared', required=True); a.add_argument('--contract-sha256', required=True); a.add_argument('--output', required=True)
    for cmd in ('verify', 'queue', 'run'):
        a = sub.add_parser(cmd); a.add_argument('--prepared', required=True); a.add_argument('--sha256', required=True)
        if cmd == 'run': a.add_argument('--output', required=True); a.add_argument('--authorization', required=True)
    a = sub.add_parser('consume'); a.add_argument('--output', required=True); a.add_argument('--sha256', required=True)
    args = parser.parse_args(argv)
    if args.cmd == 'prepare': result = write_new(args.output, material(args.contract_prepared, args.contract_sha256))
    elif args.cmd in ('verify', 'queue'):
        p = verify(args.prepared, args.sha256)
        result = dict(status='verified_not_executed', queue=p['panel']['queue'] if args.cmd == 'queue' else 4, profile=p['profile'], **FLAGS)
    elif args.cmd == 'run': result = run(args.prepared, args.sha256, args.output, args.authorization)
    else: result = consume(args.output, args.sha256)
    print(canonical(result).decode())


if __name__ == '__main__': main()
