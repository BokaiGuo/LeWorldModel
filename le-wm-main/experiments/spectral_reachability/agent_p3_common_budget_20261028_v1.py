"""Equal predict-API/sample root-only control with fully paid independent J.

Prepare requires TWO real, independently executed sealed local-signal panels.
Only MAIN's explicit run loads HOST CUDA; all other operations are saved-only.
No outcome tuning, padded calls, child signals, or scientific admission.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
import os
from pathlib import Path
import time

import numpy as np

from experiments.spectral_reachability import agent_p3_native_program_20261028_v1 as native

signal, core = native.signal, native.core
require, canonical, digest, sha = native.require, native.canonical, native.digest, native.sha
ROOT = native.ROOT
NAME = 'agent_p3_common_budget_20261028_v1'
FLAGS = dict(native.FLAGS)
MODES = ('fixed', 'adaptive')
PREVIOUS = ROOT/'results/agent_p3_native_program_20261028_v1_cuda_MAIN01/report.json'
PREVIOUS_SHA = '7f5f16ffe9d440a5cd2731e1e87c2325b1b39de9c18508ebdcb28409cbe858d3'


def rule():
    # 24 = inherited four candidates * two independent replays * max H(3).
    return dict(schema=NAME+'.rule', signal_predict_per_task_method=8,
        search_predict_per_task_method=24, total_predict_per_task_method=32,
        budget_derivation='4 inherited minimum candidates * 2 replays * max frozen horizon 3',
        horizons=[1, 3], candidates_by_horizon={'1': 12, '3': 4},
        signal='independent complete H1/H3 nominal+AD panel per method; no shared half-cost',
        scheduler='unchanged core.choose, chi=.5/gain=.05; retain original decision including four candidates',
        budget_wrapper='new explicit root search quota, not an unchanged four-candidate runtime queue',
        generator=native.rule(), fraction_rule='i/(count-1), count=24/(2*H); cast once raw float32',
        selection='minimum original Q terminal mean-square, then candidate index',
        replay='whole H then independent segmented H for every candidate; every boundary identical',
        same_predict_API_and_samples=True, equal_FLOPs_or_wall_claim=False,
        root_only=True, executed_extension_depth=1, child_signal_adapter=False,
        comparison_scope='matched predict search budget, controlled dual-signal pipeline; not same physical horizon or closed-loop success',
        fixed_H1_cost_minimal_baseline=False,
        no_padding=True, no_retry=True, no_outcome_input=True,
        unsupported_type='block entire comparison; never replace reconfiguration with progress')


def own_files():
    return [Path(__file__).resolve(), ROOT/f'tests/{NAME}_test.py', ROOT/f'innerdocs/{NAME}.md']


def exclusive(path):
    p = Path(path).absolute()
    require(p.parent == ROOT/'results' and p.name.startswith(NAME) and not p.exists() and not p.is_symlink(), 'new exclusive own result')
    return p


def write_new(path, value):
    p = exclusive(path); raw = canonical(value)
    require(len(raw) <= 16*1024**2, 'bounded metadata')
    with p.open('xb') as f:
        require(f.write(raw) == len(raw), 'complete metadata write'); f.flush(); os.fsync(f.fileno())
    signal.fsync_dir(p.parent); return dict(path=str(p), sha256=sha(p))


def load_panels(inputs):
    core.keys(inputs, MODES); panels = {}; prepared = {}; receipts = {}
    for mode in MODES: core.keys(inputs[mode], ('output', 'report_sha256'))
    require(Path(inputs['fixed']['output']).resolve() != Path(inputs['adaptive']['output']).resolve() and
        inputs['fixed']['report_sha256'] != inputs['adaptive']['report_sha256'], 'two independent actual signal executions, not one receipt copied or charged twice')
    for mode in MODES:
        core.keys(inputs[mode], ('output', 'report_sha256'))
        output = Path(inputs[mode]['output']).resolve()
        receipts[mode] = signal.consume(output, inputs[mode]['report_sha256'])
        panels[mode] = signal.read(output/'report.json')
        prepared[mode] = signal.verify(panels[mode]['prepared'], panels[mode]['prepare_sha256'])
    a, b = (panels[mode] for mode in MODES)
    require(Path(inputs['fixed']['output']).resolve() != Path(inputs['adaptive']['output']).resolve() and
        inputs['fixed']['report_sha256'] != inputs['adaptive']['report_sha256'] and
        a['artifact_sha256']['events.jsonl'] != b['artifact_sha256']['events.jsonl'], 'two independent actual signal executions, not one receipt copied or charged twice')
    require(prepared['fixed']['tasks'] == prepared['adaptive']['tasks'] and
        prepared['fixed']['policy'] == prepared['adaptive']['policy'] and
        prepared['fixed']['expected_model_hash'] == prepared['adaptive']['expected_model_hash'] and
        prepared['fixed']['runtime_rule'] == prepared['adaptive']['runtime_rule'], 'identical original tasks/root/goal/Q/model/normalizer/frozen policy')
    require(len(prepared['fixed']['tasks']) == 2, 'full two-known-task denominator')
    return prepared, panels, receipts


def branches_from(tasks, panels):
    branches = []
    for slot in (0, 1):
        t = core.validate_task(tasks[slot])
        for mode in MODES:
            panel = panels[mode]
            bs = [b for b in panel['decision_rows'] if b['slot'] == slot and b['mode'] == mode]
            require(len(bs) == 1, 'one exact original root decision per method')
            b = bs[0]; q = b['queue']; d = q['decision']; h = d['horizon']
            require(q['depth'] == 0 and q['parent_context_sha256'] == digest(t['root_context']) and
                d['program_type'] == d['extension_type'] == 'progress' and d['continue_expanding'] is True,
                'only actual root progress supported; missing reconfiguration/child adapter')
            require(type(h) is int and h in (1, 3) and d['candidates'] == 4 and len(q['requests']) == 4,
                'unchanged original H1/H3 four-candidate scheduler')
            require(all(req['prefix_edges'] == req['prefix_actions'] == [] and req['parent_context'] == t['root_context']
                for req in q['requests']), 'no borrowed root signal for child prefix')
            count, remainder = divmod(rule()['search_predict_per_task_method'], 2*h)
            require(remainder == 0 and count >= 4, 'exact useful integer quota, no padding')
            responses = [r['response'] for r in panel['rows'] if r['slot'] == slot and r['status'] == 'returned']
            require([r['horizon'] for r in responses] == [1, 3], 'complete paid pair including unused horizon')
            response = next(r for r in responses if r['horizon'] == h)
            require(response['binding_sha256'] == digest(t['binding']) and response['parent_context_sha256'] == digest(t['root_context']), 'exact local J identity')
            branches.append(dict(slot=slot, mode=mode, horizon=h, candidates=count, original_decision=deepcopy(d),
                original_queue_sha256=digest(q), response=deepcopy(response),
                signal_report_sha256=None, signal_resources=None,
                requests=[core.request(t, dict(context=t['root_context']), operation='common_budget_root_progress',
                    candidate_index=k, budget_candidates=count, original_queue_sha256=digest(q),
                    original_candidate_quota=d['candidates'], horizon=h, prefix_actions=[], prefix_edges=[],
                    response_sha256=digest(response)) for k in range(count)]))
    return branches


def split_signal_resources(receipt, slot):
    ops = [e for e in receipt['resource_receipt']['events'] if e['scope'].get('slot') == slot]
    require(ops and all(e['status'] == 'returned' for e in ops), 'full paid actual root signal ledger')
    return {axis: sum(e['delta'][axis] for e in ops) for axis in core.AXES}


def jobs_from(branches):
    return [dict(slot=b['slot'], mode=b['mode'], horizon=b['horizon'], index=k,
        program_type='progress', requested_depth=b['original_decision']['program_depth_limit'], request=deepcopy(req),
        response=deepcopy(b['response'])) for b in branches for k, req in enumerate(b['requests'])]


def equal_execution_horizon(branches):
    require([(b['slot'], b['mode']) for b in branches] == [(s, mode) for s in (0, 1) for mode in MODES], 'complete paired execution horizon denominator')
    return all(branches[2*s]['horizon'] == branches[2*s+1]['horizon'] for s in (0, 1))


def material(inputs):
    prepared, panels, receipts = load_panels(inputs); p = prepared['fixed']
    tasks = p['tasks']; branches = branches_from(tasks, panels)
    for b in branches:
        b['signal_report_sha256'] = inputs[b['mode']]['report_sha256']
        b['signal_resources'] = split_signal_resources(panels[b['mode']], b['slot'])
        signal_ops = [e for e in panels[b['mode']]['resource_receipt']['events'] if e['scope'].get('slot') == b['slot']]
        b['inclusive_signal_wall_ns'] = {resource: sum(e['ended_ns']-e['started_ns'] for e in signal_ops if e['resource'] == resource)
            for resource in sorted({e['resource'] for e in signal_ops})}
        b['signal_panel_wall_seconds'] = panels[b['mode']]['wall_seconds']
        b['signal_task_total_wall_seconds'] = None  # no invented allocation of model-loading/host overhead.
        require(b['signal_resources']['predict_API'] == b['signal_resources']['predict_samples'] == 8 and
            b['signal_resources']['jacobian_API'] == 2 and b['signal_resources']['VJP_directions'] == 384, 'complete nominal/AD J cost, never free derivatives')
    jobs = jobs_from(branches)
    files = set(own_files()) | set(native.own_files()) | {PREVIOUS}
    files.add(Path(native.generator.__file__).resolve())
    files.add(ROOT/'experiments/spectral_reachability/pusht_full_packet_chart_20261027.py')
    for mode in MODES:
        files.update(map(Path, prepared[mode]['sources']))
        files.add(Path(panels[mode]['prepared']).resolve())
        files.update(Path(inputs[mode]['output']).resolve()/n for n in ('report.json', 'events.jsonl', 'binding.json'))
    require(sha(PREVIOUS) == PREVIOUS_SHA, 'preserve sealed negative/positive native predecessor; hash only, no outcome selection')
    return dict(schema=NAME+'.prepare', inputs=deepcopy(inputs), tasks=tasks, branches=branches, jobs=jobs,
        policy=p['policy'], expected_model_hash=p['expected_model_hash'], rule=rule(), profile=native.profile(jobs),
        equal_execution_horizon=equal_execution_horizon(branches), fixed_H1_cost_minimal_baseline=False,
        sources={str(f): sha(f) for f in sorted(files)}, historical_acquisition=p['history'],
        historical_signal_panels={mode: receipts[mode]['resources'] for mode in MODES},
        predecessor_report_sha256=PREVIOUS_SHA, predecessor_outcomes_read=False,
        missing_dependencies=['MAIN audit and HOST CUDA search/replay plus strict consume',
            'independent P1/applicable P2 admission and fresh cohort authority',
            'actual new child-local signals and full tree/reconfiguration adapters; physical consequences'], **FLAGS)


def verify(path, expected):
    core.hash_value(expected); require(sha(path) == expected, 'external actual prepare SHA')
    p = signal.read(path)
    require(canonical(p) == canonical(material(p['inputs'])) and sha(path) == expected, 'complete inputs/sources/queue/budget/rule rebuild')
    return p


def generate(task, response, count, ledger):
    h = response['horizon']
    require(type(count) is int and h in (1, 3) and count == 24//(2*h), 'fixed useful quota')
    base = native.generate(task, response, ledger)
    direction = np.asarray(base['solve']['params'], np.float64)/base['divisor']
    fractions = [i/(count-1) for i in range(count)]
    return dict(solve=base['solve'], divisor=base['divisor'], fractions=fractions,
        actions=[(f*direction).astype(np.float32).reshape(5*h, 2).tolist() for f in fractions],
        response_sha256=digest(response), rule=rule())


def census(jobs): return native.census(jobs)


def winner_rows(p, rows):
    out = []
    for b in p['branches']:
        group = [r for r in rows if (r['slot'], r['mode']) == (b['slot'], b['mode'])]
        require(len(group) == b['candidates'] and all(r['status'] == 'model_replayed_not_admitted' for r in group), 'all eligible candidates, no outcome filter or partial winner')
        w = min(group, key=lambda r: (r['cost'], r['index']))
        out.append(dict(slot=b['slot'], mode=b['mode'], horizon=b['horizon'], index=w['index'], cost=w['cost'],
            actions=w['actions'], terminal_context_sha256=digest(w['whole'][-1]), executed_extension_depth=1,
            requested_depth=b['original_decision']['program_depth_limit'], physical_release=False))
    return out


def panel(p, factory, ledger, store, rows):
    for b in p['branches']:
        slot, mode, h, count = (b[k] for k in ('slot', 'mode', 'horizon', 'candidates'))
        ledger.scope = dict(slot=slot, mode=mode, index=0, horizon=h)
        bank = ledger.call('program_compile', lambda: generate(p['tasks'][slot], b['response'], count, ledger))
        store.event(dict(kind='bank', slot=slot, mode=mode, bank=deepcopy(bank)))
        for job, row in [(j, r) for j, r in zip(p['jobs'], rows, strict=True) if (j['slot'], j['mode']) == (slot, mode)]:
            ledger.scope = {k: job[k] for k in ('slot', 'mode', 'index', 'horizon')}
            row['status'] = 'attempted'; store.event(dict(kind='candidate_request', job_sha256=digest(job), row=deepcopy(row)))
            try:
                row['actions'] = bank['actions'][job['index']]; backend = factory(slot)
                row['whole'] = ledger.call('replay', lambda: backend.replay(p['tasks'][slot]['root_context'], row['actions'], False, ledger))
                row['segmented'] = ledger.call('replay', lambda: backend.replay(p['tasks'][slot]['root_context'], row['actions'], True, ledger))
                require(row['whole'] == row['segmented'], 'all independent whole/segmented boundaries identical')
                terminal = core.checked_path(p['tasks'][slot], p['tasks'][slot]['root_context'], row['actions'], row['whole'])
                row['cost'] = ledger.call('metric', lambda: core.model_cost(p['tasks'][slot], terminal))
                row['status'] = 'model_replayed_not_admitted'; store.event(dict(kind='candidate_return', row=deepcopy(row)))
            except BaseException as exc:
                row.update(status='failed', error=type(exc).__name__+': '+str(exc)); raise
    return winner_rows(p, rows)


def accounting(p, receipt, rows):
    core.audit_ledger(receipt); branches = []
    require(receipt['caps'] == receipt['attempted'] == receipt['completed'] == p['profile']['resource_caps'], 'complete new search all-axis actual cost')
    for b in p['branches']:
        ops = [e for e in receipt['events'] if (e['scope']['slot'], e['scope']['mode']) == (b['slot'], b['mode'])]
        search = {a: sum(e['delta'][a] for e in ops) for a in core.AXES}
        signal_cost = b['signal_resources']; total = {a: search[a]+signal_cost[a] for a in core.AXES}
        require(search['predict_API'] == search['predict_samples'] == 24 and signal_cost['predict_API'] == signal_cost['predict_samples'] == 8 and
            total['predict_API'] == total['predict_samples'] == 32, 'actual equal total API AND samples including full paid J')
        require(search['replay_calls'] == 2*b['candidates'] and search['metric_calls'] == b['candidates'], 'all real useful candidates, no padding')
        group = [r for r in rows if (r['slot'], r['mode']) == (b['slot'], b['mode'])]
        branches.append(dict(slot=b['slot'], mode=b['mode'], horizon=b['horizon'], candidates=b['candidates'],
            execution_horizon_packets=b['horizon'], raw_action_steps=5*b['horizon'],
            signal_resources=signal_cost, search_resources=search, total_resources=total,
            inclusive_signal_wall_ns=b['inclusive_signal_wall_ns'], signal_panel_wall_seconds=b['signal_panel_wall_seconds'],
            signal_task_total_wall_seconds=b['signal_task_total_wall_seconds'],
            all_modelQ=[r['cost'] for r in group], distinct_actions=len({digest(r['actions']) for r in group}),
            inclusive_search_wall_ns={resource: sum(e['ended_ns']-e['started_ns'] for e in ops if e['resource'] == resource)
                for resource in sorted({e['resource'] for e in ops})}))
    return branches


def run(prepared, expected, output, authorization):
    require(authorization == 'MAIN_APPROVED_HOST_CUDA', 'MAIN source review and explicit HOST CUDA required')
    started = time.monotonic(); p = verify(prepared, expected); output = exclusive(output)
    store = signal.Store(output, p['profile']); store.started = started
    ledger = core.Ledger(p['profile']['resource_caps'], sink=store.event); rows = census(p['jobs'])
    errors = []; model = None; before = after = host = None; winners = []; resources = None; sources_ok = False
    try:
        store.put('binding.json', dict(prepared=str(Path(prepared).resolve()), prepare_sha256=expected))
        store.event(dict(kind='denominator', rows=deepcopy(rows)))
        os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host = require_cuda_allocation('cuda:0'); device = torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        store.memory = lambda: (torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device))
        store.check(); model = load_compatible_pretrained(p['policy'], device=device).eval().requires_grad_(False)
        before = signal.model_hash(model); require(before['original_state_dict'] == p['expected_model_hash'], 'original frozen model')
        store.event(dict(kind='model_before', hashes=before))
        for module in model.modules():
            require(str(Path(inspect.getfile(type(module))).resolve()) in p['sources'], 'presealed native class closure')
        adapters = [native.NativeProgram(model, device, t, store) for t in p['tasks']]
        with torch.autocast(device_type='cuda', enabled=False), sdpa_kernel(SDPBackend.MATH):
            winners = panel(p, lambda s: adapters[s], ledger, store, rows)
        torch.cuda.synchronize(device); store.check(); resources = accounting(p, ledger.receipt(), rows)
    except BaseException as exc:
        errors.append(dict(stage='execution', message=type(exc).__name__+': '+str(exc)))
    finally:
        for row in rows:
            if row['status'] == 'not_started': row['status'] = 'blocked_not_dispatched'
            elif row['status'] == 'attempted': row['status'] = 'failed'
        if model is not None:
            try:
                after = signal.model_hash(model); require(before == after, 'unchanged frozen state/all buffers')
                store.event(dict(kind='model_after', hashes=after))
            except BaseException as exc: errors.append(dict(stage='model_final', message=str(exc)))
        try:
            verify(prepared, expected); store.check(); sources_ok = True
            store.event(dict(kind='sources_final', sha256=digest(p['sources'])))
        except BaseException as exc: errors.append(dict(stage='sources_or_limits_final', message=str(exc)))
        store.close(); complete = not errors and sources_ok and len(winners) == 4 and resources is not None
        report = dict(schema=NAME+'.report', status='equal_predict_saved_not_admitted' if complete else 'partial_or_failed',
            equal_execution_horizon=equal_execution_horizon(p['branches']), fixed_H1_cost_minimal_baseline=False,
            same_closed_loop_success_claim=False, comparison_scope=rule()['comparison_scope'],
            prepared=str(Path(prepared).resolve()), prepare_sha256=expected, rows=rows, winners=winners, per_branch=resources,
            errors=errors, resource_receipt=ledger.receipt(), model_before=before, model_after=after,
            sources_verified=sources_ok, source_manifest_sha256=digest(p['sources']), host=host,
            saved_verification=dict(upstream_saved_SVD_calls=16 if sources_ok else None,
                semantics='two upstream panels *4 SVD *pre/post verify; partial work unknown'),
            wall_seconds=store.clock()-store.started, CUDA_peak_bytes=store.memory() if store.memory else None,
            journal_tail=store.tail, artifact_sha256={n: sha(output/n) for n in ('binding.json', 'events.jsonl')}, **FLAGS)
        store.put('report.json', report)
    require(complete, 'partial/failed retained with full denominator, no retry: '+repr(errors))
    return dict(output=str(output), report_sha256=sha(output/'report.json'))


def validate_saved(p, r, events):
    require(type(r['equal_execution_horizon']) is bool and r['equal_execution_horizon'] == equal_execution_horizon(p['branches']) and
        r['fixed_H1_cost_minimal_baseline'] is False and r['same_closed_loop_success_claim'] is False and
        r['comparison_scope'] == rule()['comparison_scope'], 'explicit unequal execution horizon and controlled-pipeline baseline scope, not physical/closed-loop success')
    require(len(r['rows']) == len(p['jobs']) and len(p['branches']) == 4, 'complete predeclared denominator')
    require(events[0] == dict(kind='denominator', rows=census(p['jobs'])) and len([e for e in events if e['kind'] == 'denominator']) == 1, 'pre-dispatch full denominator')
    allowed = {'denominator', 'bank', 'candidate_request', 'candidate_return', 'model_before', 'model_after', 'sources_final',
        'resource_attempt', 'resource_return', 'io_input', 'io_return', 'program_packet', 'program_boundary'}
    require(all(e['kind'] in allowed for e in events), 'no unknown/failed/outcome events')
    for e in events:
        fields = ('kind', 'entry') if e['kind'].startswith('resource_') else {
            'denominator': ('kind', 'rows'), 'bank': ('kind', 'slot', 'mode', 'bank'),
            'candidate_request': ('kind', 'job_sha256', 'row'), 'candidate_return': ('kind', 'row'),
            'model_before': ('kind', 'hashes'), 'model_after': ('kind', 'hashes'), 'sources_final': ('kind', 'sha256'),
            'io_input': ('kind', 'scope', 'phase', 'step', 'operation_index', 'arrays'),
            'io_return': ('kind', 'scope', 'phase', 'step', 'operation_index', 'arrays'),
            'program_packet': ('kind', 'scope', 'phase', 'step', 'operation_index', 'arrays'),
            'program_boundary': ('kind', 'scope', 'phase', 'step', 'operation_index', 'arrays')}[e['kind']]
        core.keys(e, fields)
    require([e for e in events if e['kind'] == 'candidate_request'] == [dict(kind='candidate_request', job_sha256=digest(j), row=dict(row, status='attempted'))
        for j, row in zip(p['jobs'], census(p['jobs']), strict=True)], 'all exact real search requests consumed')
    require([e['row'] for e in events if e['kind'] == 'candidate_return'] == r['rows'], 'all durable candidate results')
    banks = []
    for b in p['branches']:
        ledger = core.Ledger(p['profile']['resource_caps']); ledger.scope = dict(saved_recompute=True)
        banks.append(dict(kind='bank', slot=b['slot'], mode=b['mode'], bank=generate(p['tasks'][b['slot']], b['response'], b['candidates'], ledger)))
    require([e for e in events if e['kind'] == 'bank'] == banks, 'fixed generator/budget/action recomputation')
    for job, row in zip(p['jobs'], r['rows'], strict=True):
        require(all(row[k] == job[k] for k in ('slot', 'mode', 'index', 'horizon')) and row['status'] == 'model_replayed_not_admitted' and row['error'] is None, 'full original identity and no missing zero-fill')
        bank = next(e['bank'] for e in banks if (e['slot'], e['mode']) == (row['slot'], row['mode']))
        require(row['actions'] == bank['actions'][row['index']] and row['whole'] == row['segmented'], 'all sealed actions/double replay exact')
        t = p['tasks'][row['slot']]; terminal = core.checked_path(t, t['root_context'], row['actions'], row['whole'])
        require(type(row['cost']) is float and row['cost'] == core.model_cost(t, terminal), 'original Q task-only objective')
    require(r['winners'] == winner_rows(p, r['rows']), 'stable complete model-only winner')
    require(r['per_branch'] == accounting(p, r['resource_receipt'], r['rows']), 'equal actual API/sample costs and all distinct axes')
    native.validate_arrays(p, r['rows'], events, r['resource_receipt'])


def consume(output, expected):
    output = Path(output).absolute(); core.hash_value(expected)
    require(sha(output/'report.json') == expected, 'external actual report SHA')
    r = signal.read(output/'report.json'); p = verify(r['prepared'], r['prepare_sha256'])
    core.keys(r, ('schema', 'status', 'prepared', 'prepare_sha256', 'rows', 'winners', 'per_branch', 'errors', 'resource_receipt',
        'equal_execution_horizon', 'fixed_H1_cost_minimal_baseline', 'same_closed_loop_success_claim', 'comparison_scope',
        'model_before', 'model_after', 'sources_verified', 'source_manifest_sha256', 'host', 'saved_verification',
        'wall_seconds', 'CUDA_peak_bytes', 'journal_tail', 'artifact_sha256', *FLAGS))
    require(r['schema'] == NAME+'.report' and r['status'] == 'equal_predict_saved_not_admitted' and not r['errors'], 'complete engineering result only')
    require(all(r[k] is False for k in FLAGS) and r['sources_verified'] is True and r['source_manifest_sha256'] == digest(p['sources']), 'no admission/source changes')
    require(r['model_before'] == r['model_after'] and r['model_before']['original_state_dict'] == p['expected_model_hash'], 'frozen model and buffers')
    require(r['saved_verification'] == dict(upstream_saved_SVD_calls=16, semantics='two upstream panels *4 SVD *pre/post verify; partial work unknown'), 'separate actual saved verification CPU work')
    require({f.name for f in output.iterdir()} == {'report.json', 'binding.json', 'events.jsonl'} and set(r['artifact_sha256']) == {'binding.json', 'events.jsonl'}, 'exact artifact set')
    require(signal.read(output/'binding.json') == dict(prepared=r['prepared'], prepare_sha256=r['prepare_sha256']), 'execution binding')
    for n, h in r['artifact_sha256'].items(): require(sha(output/n) == h, 'complete saved artifact hashes')
    events, tail = signal.journal(output/'events.jsonl'); require(tail == r['journal_tail'], 'journalchain/report binding')
    require([e['hashes'] for e in events if e['kind'] == 'model_before'] == [r['model_before']] and
        [e['hashes'] for e in events if e['kind'] == 'model_after'] == [r['model_after']] and
        [e['sha256'] for e in events if e['kind'] == 'sources_final'] == [digest(p['sources'])], 'actual before/after journal')
    require(type(r['wall_seconds']) in (float, int) and 0 <= r['wall_seconds'] <= p['profile']['wall_seconds'], 'actual wall cap')
    require(type(r['CUDA_peak_bytes']) is list and len(r['CUDA_peak_bytes']) == 2 and all(type(v) is int and 0 <= v <= p['profile'][key]
        for v, key in zip(r['CUDA_peak_bytes'], ('cuda_allocated_bytes', 'cuda_reserved_bytes'))), 'actual memory cap')
    require(sum(f.stat().st_size for f in output.iterdir()) <= p['profile']['output_bytes'], 'actual output cap')
    validate_saved(p, r, events)
    require(sha(output/'report.json') == expected and all(sha(f) == h for f, h in p['sources'].items()) and
        all(sha(output/n) == h for n, h in r['artifact_sha256'].items()), 'unchanged full source/artifact closure after consume')
    return dict(schema=NAME+'.consume', status='equal_predict_saved_verified_not_admitted', report_sha256=expected,
        equal_execution_horizon=r['equal_execution_horizon'], fixed_H1_cost_minimal_baseline=False,
        same_closed_loop_success_claim=False, comparison_scope=rule()['comparison_scope'],
        per_branch=r['per_branch'], winners=r['winners'], denominator=len(p['jobs']),
        backward_engine_calls=None, CUDA_ns=None, FLOPs=None, same_wall_or_total_compute_claim=False,
        saved_verification_array_SVD_calls=12, saved_verification_semantics='two upstream consumers 8 + four generator recomputations 4',
        acquisition_unknowns=p['historical_acquisition'], scientific_effect=None, complete_tree=False, physical_benefit=None, **FLAGS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('plan')
    a = sub.add_parser('prepare'); a.add_argument('--output', required=True)
    for mode in MODES:
        a.add_argument('--'+mode+'-signal-output', required=True); a.add_argument('--'+mode+'-signal-sha256', required=True)
    for cmd in ('verify', 'queue', 'run'):
        a = sub.add_parser(cmd); a.add_argument('--prepared', required=True); a.add_argument('--sha256', required=True)
        if cmd == 'run': a.add_argument('--output', required=True); a.add_argument('--authorization', required=True)
    a = sub.add_parser('consume'); a.add_argument('--output', required=True); a.add_argument('--sha256', required=True)
    args = parser.parse_args(argv)
    if args.cmd == 'plan': result = dict(rule=rule(), required_independent_signal_reports=2, neural_execution=False, **FLAGS)
    elif args.cmd == 'prepare':
        inputs = {mode: dict(output=str(Path(getattr(args, mode+'_signal_output')).resolve()), report_sha256=getattr(args, mode+'_signal_sha256')) for mode in MODES}
        result = write_new(args.output, material(inputs))
    elif args.cmd in ('verify', 'queue'):
        p = verify(args.prepared, args.sha256)
        result = dict(status='verified_not_executed', branches=p['branches'], jobs=p['jobs'] if args.cmd == 'queue' else len(p['jobs']), **FLAGS)
    elif args.cmd == 'run': result = run(args.prepared, args.sha256, args.output, args.authorization)
    else: result = consume(args.output, args.sha256)
    print(canonical(result).decode())


if __name__ == '__main__': main()
