"""Root-only saved-J progress extensions; only explicit MAIN run loads CUDA.

No child derivative, physical release, fitting, or full-C3 assertion. Saved
consume checks native buffer wiring, not an independent model recomputation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
import os
from pathlib import Path
import time

import numpy as np

from experiments.spectral_reachability import agent_p3_local_signal_cuda_20261028_v1 as signal
from experiments.spectral_reachability import pusht_full_packet_compiler_20261027 as generator

NAME = 'agent_p3_native_program_20261028_v1'
ROOT = signal.ROOT
REPORT = ROOT/'results/agent_p3_local_signal_cuda_20261028_v1_cuda_MAIN02/report.json'
REPORT_SHA = '5c776ea604f615081b9f50b92648e22e2612ef46d8443bf1949f69341ed4864d'
FLAGS = dict(signal.FLAGS, scientific_admission=False, execution_authorized=False)
core = signal.core
require, canonical, digest, sha = signal.require, signal.canonical, signal.digest, signal.sha


def rule():
    return dict(generator='pusht_full_packet_compiler_20261027.bounded_ridge',
        damping=generator.DAMPING, joint_raw_l2_cap=generator.CAP,
        fractions=[0., 1/3, 2/3, 1.],
        coordinates='saved J is d latent/d raw at zero; NOT logits',
        action_map='float64 ridge / max(1,max(abs(ridge))), then fraction, cast once float32',
        candidates=4, selection='minimum original Q terminal mean-square, stable candidate index',
        replay='each candidate whole H then independently segmented H; compare every boundary exactly',
        root_only=True, child_signals_reused=False, depth_is_not_executed=True,
        reconfiguration='unsupported: fail closed, no surrogate objective',
        outcome_input=False, equal_compute_claim=False, no_padding=True)


def own_files():
    return [Path(__file__).resolve(), ROOT/f'tests/{NAME}_test.py', ROOT/f'innerdocs/{NAME}.md']


def exclusive(path):
    p = Path(path).absolute()
    require(p.parent == ROOT/'results' and p.name.startswith(NAME) and not p.exists() and not p.is_symlink(),
        'new exclusive own result required')
    return p


def source_panel(report=REPORT, expected=REPORT_SHA):
    # The sealed upstream consumer is mandatory, not merely a status check.
    report = Path(report).resolve()
    require(report == REPORT and expected == REPORT_SHA, 'fixed original two-task signal report')
    receipt = signal.consume(report.parent, expected)
    r = signal.read(report); p = signal.verify(r['prepared'], r['prepare_sha256'])
    require(receipt['denominator'] == 4 and receipt['decision_denominator'] == 4, 'full paid source denominator')
    return p, r, receipt


def jobs_from(p, report):
    require(len(p['tasks']) == 2 and len(report['decision_rows']) == 4, 'two original known tasks, both methods')
    jobs = []
    for i, branch in enumerate(report['decision_rows']):
        slot, mode = divmod(i, 2); mode = ('fixed', 'adaptive')[mode]
        require(branch['slot'] == slot and branch['mode'] == mode, 'slot-major fixed/adaptive identity')
        task = core.validate_task(p['tasks'][slot]); q = branch['queue']; decision = q['decision']
        require(q['depth'] == 0 and q['parent_context_sha256'] == digest(task['root_context']), 'original root only')
        require(decision['candidates'] == 4 and len(q['requests']) == 4 and decision['horizon'] in (1, 3), 'fixed four-candidate H1/H3 denominator')
        h = decision['horizon']
        matches = [r['response'] for r in report['rows'] if r['slot'] == slot and r['horizon'] == h and r['status'] == 'returned']
        require(len(matches) == 1, 'one actual saved same-node response')
        response = matches[0]
        require(response['binding_sha256'] == digest(task['binding']) and response['parent_context_sha256'] == digest(task['root_context']), 'no cross-root or child reuse')
        for k in range(4):
            jobs.append(dict(slot=slot, mode=mode, index=k, horizon=h,
                program_type=decision['program_type'], requested_depth=decision['program_depth_limit'],
                request=deepcopy(q['requests'][k]), response=deepcopy(response)))
    return jobs


def profile(jobs):
    # Each real whole/segmented path pays H transitions. No padding to H3.
    n = sum(2*j['horizon'] for j in jobs if j['program_type'] == 'progress')
    groups = len({(j['slot'], j['mode']) for j in jobs if j['program_type'] == 'progress'})
    caps = dict.fromkeys(core.AXES, 0)
    caps.update(predict_API=n, predict_samples=n, predict_context_tokens=3*n,
        action_encoder_calls=n, action_encoder_samples=n, action_encoder_context_tokens=3*n,
        predictor_calls=n, predictor_samples=n, predictor_context_tokens=3*n,
        pred_proj_calls=n, pred_proj_rows=3*n, array_SVD_calls=groups, solve_calls=groups,
        program_compile_calls=groups, replay_calls=2*sum(j['program_type'] == 'progress' for j in jobs),
        metric_calls=sum(j['program_type'] == 'progress' for j in jobs))
    return dict(resource_caps=caps, wall_seconds=900, output_bytes=256*1024**2,
        final_reserve_bytes=16*1024**2, disk_floor_bytes=200*1024**2,
        cuda_allocated_bytes=3*1024**3, cuda_reserved_bytes=4*1024**3,
        CPU_neural=False, TF32=False, SDP='MATH', no_retry=True, no_padding=True)


def material(report=REPORT, expected=REPORT_SHA):
    p, r, receipt = source_panel(report, expected); jobs = jobs_from(p, r)
    files = set(map(Path, p['sources'])) | set(own_files()) | {Path(generator.__file__).resolve()}
    files.add(ROOT/'experiments/spectral_reachability/pusht_full_packet_chart_20261027.py')
    files |= {Path(report).resolve(), Path(r['prepared']).resolve(), Path(report).parent/'events.jsonl', Path(report).parent/'binding.json'}
    return dict(schema=NAME+'.prepare', report=str(Path(report).resolve()), report_sha256=expected,
        signal_prepared=r['prepared'], signal_prepare_sha256=r['prepare_sha256'], tasks=p['tasks'],
        policy=p['policy'], expected_model_hash=p['expected_model_hash'], jobs=jobs, rule=rule(),
        profile=profile(jobs), sources={str(f): sha(f) for f in sorted(files)},
        historical_paid_resources=receipt['resources'], historical_acquisition=p['history'],
        signal_reacquired=False, total_pipeline_equivalence=False,
        scope='two known root nodes; one extension per task/method; no descendant expansion', **FLAGS)


def write_new(path, value):
    p = exclusive(path); raw = canonical(value)
    require(len(raw) <= 16*1024**2, 'bounded prepare')
    with p.open('xb') as f:
        require(f.write(raw) == len(raw), 'short prepare write'); f.flush(); os.fsync(f.fileno())
    signal.fsync_dir(p.parent)
    return dict(path=str(p), sha256=sha(p))


def verify(path, expected):
    core.hash_value(expected); require(sha(path) == expected, 'external prepare SHA')
    p = signal.read(path)
    require(canonical(p) == canonical(material(p['report'], p['report_sha256'])) and sha(path) == expected,
        'exact source/task/saved signal/generator/rule/resource closure')
    return p


def generate(task, response, ledger):
    """Reuse the actual bounded ridge solver; it performs exactly one SVD.

    No global SVD monkeypatch. The outer solve and inner array_svd debit the
    concrete helper invocation once; its fixed source is sealed by prepare.
    """
    task = core.validate_task(task); h = response['horizon']
    require(type(h) is int and h in (1, 3) and response['binding_sha256'] == digest(task['binding']) and
        response['parent_context_sha256'] == digest(task['root_context']), 'same actual root derivative only')
    j = core.finite(response['jacobian'], (192, 10*h), True)
    nominal = core.finite(response['nominal'], (192,), True)
    q = np.asarray(task['projection'], np.float64); residual = q@(np.asarray(task['goal'])-nominal)
    solve = ledger.call('solve', lambda: ledger.call('array_svd', lambda: generator.bounded_ridge(q@j, residual,
        damping=rule()['damping'], cap=rule()['joint_raw_l2_cap'])))
    direction = np.asarray(solve['params']); divisor = max(1., float(np.max(np.abs(direction))))
    actions = [(fraction*direction/divisor).astype(np.float32).reshape(5*h, 2).tolist() for fraction in rule()['fractions']]
    return dict(solve=solve, divisor=divisor, actions=actions, response_sha256=digest(response), rule=rule())


def census(jobs):
    return [dict(slot=j['slot'], mode=j['mode'], index=j['index'], horizon=j['horizon'],
        status='not_started', actions=None, whole=None, segmented=None, cost=None, error=None) for j in jobs]


def panel(p, factory, ledger, store, rows):
    """Same denominator loop for production CUDA and explicit RAM fixtures."""
    banks = {}; winners = []
    for job, row in zip(p['jobs'], rows, strict=True):
        slot, mode, k, h = (job[x] for x in ('slot', 'mode', 'index', 'horizon'))
        ledger.scope = dict(slot=slot, mode=mode, index=k, horizon=h)
        if job['program_type'] != 'progress':
            row.update(status='blocked_unsupported_program_type', error='no real reconfiguration/composition adapter')
            store.event(dict(kind='candidate_blocked', row=deepcopy(row))); continue
        task = p['tasks'][slot]; row['status'] = 'attempted'
        store.event(dict(kind='candidate_request', job_sha256=digest(job), row=deepcopy(row)))
        try:
            key = (slot, mode)
            if key not in banks:
                banks[key] = ledger.call('program_compile', lambda: generate(task, job['response'], ledger))
                store.event(dict(kind='bank', slot=slot, mode=mode, bank=deepcopy(banks[key])))
            row['actions'] = banks[key]['actions'][k]
            backend = factory(slot)
            row['whole'] = ledger.call('replay', lambda: backend.replay(task['root_context'], row['actions'], False, ledger))
            row['segmented'] = ledger.call('replay', lambda: backend.replay(task['root_context'], row['actions'], True, ledger))
            require(row['whole'] == row['segmented'], 'all whole/segmented packet boundaries exactly equal')
            terminal = core.checked_path(task, task['root_context'], row['actions'], row['whole'])
            residual = np.asarray(task['projection'])@(np.asarray(terminal['latent_history'][-1])-np.asarray(task['goal']))
            row['cost'] = ledger.call('metric', lambda: float(np.mean(residual**2)))
            require(np.isfinite(row['cost']), 'finite original Q terminal cost')
            row['status'] = 'model_replayed_not_admitted'
            store.event(dict(kind='candidate_return', row=deepcopy(row)))
        except BaseException as exc:
            row.update(status='failed', error=type(exc).__name__+': '+str(exc)); raise
    for slot in (0, 1):
        for mode in ('fixed', 'adaptive'):
            group = [r for r in rows if r['slot'] == slot and r['mode'] == mode]
            if len(group) == 4 and all(r['status'] == 'model_replayed_not_admitted' for r in group):
                w = min(group, key=lambda r: (r['cost'], r['index']))
                winners.append(dict(slot=slot, mode=mode, index=w['index'], horizon=w['horizon'], cost=w['cost'],
                    actions=w['actions'], terminal_context_sha256=digest(w['whole'][-1]),
                    requested_depth=next(j['requested_depth'] for j in p['jobs'] if j['slot'] == slot and j['mode'] == mode),
                    executed_extension_depth=1, physical_release=False))
    return winners


class NativeProgram(signal.NativeSignal):
    """Instance-local real native hooks inherited unchanged from sealed signal.

    Replays use full rolling latent3/action2 ABI. Segmented independently
    reconstructs CUDA tensors from exact saved float32 context each packet.
    """
    def replay(self, parent, actions, segmented, ledger):
        core.context(parent); core.finite(actions, (len(actions), 2), True)
        require(len(actions) in (5, 15) and np.max(np.abs(actions)) <= 1, 'raw H1/H3 actions')
        require(parent == self.task['root_context'], 'only root invocation; no borrowed child signal')
        import torch
        self.frozen(); self.ledger = ledger; self.phase = 'segmented' if segmented else 'whole'
        state = torch.tensor(parent['latent_history'], dtype=torch.float32, device=self.device)[None]
        history = torch.tensor(parent['action_history'], dtype=torch.float32, device=self.device)[None]
        mean = torch.tensor(self.task['action_mean'], dtype=torch.float32, device=self.device)
        std = torch.tensor(self.task['action_std'], dtype=torch.float32, device=self.device)
        path = []
        with torch.no_grad(), self.observe():
            for step, packet in enumerate(np.asarray(actions, np.float32).reshape(-1, 5, 2)):
                self.step = step
                if segmented and path:
                    state = torch.tensor(path[-1]['latent_history'], dtype=torch.float32, device=self.device)[None]
                    history = torch.tensor(path[-1]['action_history'], dtype=torch.float32, device=self.device)[None]
                raw = torch.tensor(packet, dtype=torch.float32, device=self.device)
                token = ((raw-mean)/std).reshape(10); outgoing = torch.cat((history, token[None, None]), dim=1)
                self.arrays('program_packet', latent=state, history=history, raw=raw, mean=mean, std=std, token=token, outgoing=outgoing)
                encoded = self.model.action_encoder(outgoing)
                predicted = self.call('predict', lambda: self.model.predict(state, encoded),
                    dict(latent=state, act_emb=encoded), shape=list(state.shape))
                require(tuple(predicted.shape) == (1, 3, 192), 'actual native full prediction ABI')
                state = torch.cat((state[:, 1:], predicted[:, -1:]), dim=1); history = outgoing[:, 1:]
                self.arrays('program_boundary', latent=state, history=history)
                after = dict(parent, latent_history=state[0].cpu().tolist(), action_history=history[0].cpu().tolist())
                path.append(after)
        self.frozen(); core.checked_path(self.task, parent, actions, path); return path


def run(prepared, expected, output, authorization):
    require(authorization == 'MAIN_APPROVED_HOST_CUDA', 'MAIN explicit HOST CUDA authorization required')
    started = time.monotonic(); p = verify(prepared, expected); output = exclusive(output)
    store = signal.Store(output, p['profile']); store.started = started
    ledger = core.Ledger(p['profile']['resource_caps'], sink=store.event); rows = census(p['jobs'])
    errors = []; model = None; before = after = host = None; winners = []; source_ok = False
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
            require(str(Path(inspect.getfile(type(module))).resolve()) in p['sources'], 'presealed native class source')
        adapters = [NativeProgram(model, device, task, store) for task in p['tasks']]
        with torch.autocast(device_type='cuda', enabled=False), sdpa_kernel(SDPBackend.MATH):
            winners = panel(p, lambda s: adapters[s], ledger, store, rows)
        torch.cuda.synchronize(device); store.check()
        require(len(winners) == 4 and ledger.attempted == ledger.completed == p['profile']['resource_caps'], 'complete exact native denominator/resources')
    except BaseException as exc:
        errors.append(dict(stage='execution', error=type(exc).__name__+': '+str(exc)))
    finally:
        for row in rows:
            if row['status'] == 'not_started': row['status'] = 'blocked_not_dispatched'
            elif row['status'] == 'attempted': row['status'] = 'failed'
        if model is not None:
            try:
                after = signal.model_hash(model); require(before == after, 'unchanged frozen model including all buffers')
                store.event(dict(kind='model_after', hashes=after))
            except BaseException as exc: errors.append(dict(stage='model_final', error=str(exc)))
        try:
            verify(prepared, expected); source_ok = True
            store.event(dict(kind='sources_final', sha256=digest(p['sources'])))
            store.check()
        except BaseException as exc: errors.append(dict(stage='sources_or_limits_final', error=str(exc)))
        store.close()
        complete = not errors and source_ok and len(winners) == 4
        report = dict(schema=NAME+'.report', status='root_extensions_saved_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).resolve()), prepare_sha256=expected, rows=rows, winners=winners,
            errors=errors, resource_receipt=ledger.receipt(), model_before=before, model_after=after,
            source_manifest_sha256=digest(p['sources']), sources_verified=source_ok,
            saved_verification=dict(completed_upstream_passes=2 if source_ok else 1,
                upstream_array_SVD_calls=8 if source_ok else None,
                partial_failed_pass_work_unknown=not source_ok, outside_native_generation_ledger=True),
            host=host, wall_seconds=store.clock()-store.started, CUDA_peak_bytes=store.memory() if store.memory else None,
            journal_tail=store.tail, artifact_sha256={n: sha(output/n) for n in ('events.jsonl', 'binding.json')}, **FLAGS)
        store.put('report.json', report)
    require(complete, 'partial/failed retained; no retry: '+repr(errors))
    return dict(output=str(output), report_sha256=sha(output/'report.json'))


def validate_arrays(p, rows, events, receipt):
    """Pure saved arithmetic closure from actual IO; never executes a model."""
    operations = receipt['events']; resource = [e for e in events if e['kind'].startswith('resource_')]
    expected = []
    for op in operations:
        require(op['status'] == 'returned', 'all native attempts returned')
        expected += [dict(kind='resource_attempt', entry=dict(op, status='attempted', ended_ns=None)), dict(kind='resource_return', entry=op)]
    require(sorted(map(canonical, resource)) == sorted(map(canonical, expected)), 'journal/ledger actual attempt-return bijection')
    stack = []
    for e in resource:
        if e['kind'] == 'resource_attempt': stack.append(e['entry']['index'])
        else: require(stack and stack.pop() == e['entry']['index'], 'nested native attempt/return order')
    require(not stack, 'no dangling actual native calls')
    io = [e for e in events if e['kind'] in ('io_input', 'io_return')]
    native_ops = [o for o in operations if o['resource'] in ('predict', 'action_encoder', 'predictor', 'pred_proj')]
    require(len(io) == 2*len(native_ops), 'complete all-and-only native buffers')
    for op in native_ops:
        linked = [e for e in io if e['operation_index'] == op['index']]
        require([e['kind'] for e in linked] == ['io_input', 'io_return'] and all(e['scope'] == op['scope'] for e in linked), 'native buffers linked to paid operation')
        inputs = linked[0]['arrays']
        first = {'predict': 'predict:latent', 'action_encoder': 'action_encoder:0',
            'predictor': 'predictor:0', 'pred_proj': 'pred_proj:0'}[op['resource']]
        require(list(signal.unblob(inputs[first]).shape) == op['shape'], 'paid shape matches actual native input buffer')
    for row in rows:
        task = p['tasks'][row['slot']]; h = row['horizon']
        scope = {k: row[k] for k in ('slot', 'mode', 'index', 'horizon')}
        for phase in ('whole', 'segmented'):
            local = [e for e in events if e.get('scope') == scope and e.get('phase') == phase]
            packets = [e for e in local if e['kind'] == 'program_packet']
            boundaries = [e for e in local if e['kind'] == 'program_boundary']
            require(len(packets) == len(boundaries) == h and len([e for e in local if e['kind'] == 'io_input']) == 4*h, 'exact paid packet/module denominator')
            before = task['root_context']
            for step, (pe, be, after) in enumerate(zip(packets, boundaries, row[phase], strict=True)):
                require(pe['step'] == be['step'] == step, 'ordered full packet boundaries')
                packet = {k: signal.unblob(v) for k, v in pe['arrays'].items()}
                boundary = {k: signal.unblob(v) for k, v in be['arrays'].items()}
                require(set(packet) == {'latent', 'history', 'raw', 'mean', 'std', 'token', 'outgoing'} and set(boundary) == {'latent', 'history'}, 'exact native buffer names')
                require(all(a.dtype == np.float32 for a in [*packet.values(), *boundary.values()]), 'actual float32 native precision')
                raw = np.asarray(row['actions'][5*step:5*step+5], np.float32)
                mean, std = (np.asarray(task[k], np.float32) for k in ('action_mean', 'action_std'))
                latent = np.asarray(before['latent_history'], np.float32)[None]; history = np.asarray(before['action_history'], np.float32)[None]
                token = ((raw-mean)/std).reshape(10); outgoing = np.concatenate((history, token[None, None]), axis=1)
                for k, value in dict(latent=latent, history=history, raw=raw, mean=mean, std=std, token=token, outgoing=outgoing).items():
                    require(np.array_equal(packet[k], value), 'exact saved raw/normalizer/parent packet: '+k)
                def get(resource, returned=False):
                    kind = 'io_return' if returned else 'io_input'
                    candidates = [e for e in local if e['kind'] == kind and e['step'] == step and any(k.startswith(resource+':') for k in e['arrays'])]
                    require(len(candidates) == 1, 'one native module IO per packet')
                    return {k: signal.unblob(v) for k, v in candidates[0]['arrays'].items()}
                ai, ao = get('action_encoder'), get('action_encoder', True)
                pi, po = get('predict'), get('predict', True)
                ni, no = get('predictor'), get('predictor', True)
                qi, qo = get('pred_proj'), get('pred_proj', True)
                require(all(a.dtype == np.float32 for group in (ai, ao, pi, po, ni, no, qi, qo) for a in group.values()), 'module IO original float32')
                require(np.array_equal(ai['action_encoder:0'], outgoing) and
                    np.array_equal(pi['predict:latent'], latent) and np.array_equal(pi['predict:act_emb'], ao['action_encoder:output']) and
                    np.array_equal(ni['predictor:0'], latent) and np.array_equal(ni['predictor:1'], ao['action_encoder:output']) and
                    np.array_equal(qi['pred_proj:0'], no['predictor:output'].reshape(3, 192)) and
                    np.array_equal(qo['pred_proj:output'].reshape(1, 3, 192), po['predict:output']), 'actual AE/predict/predictor/pred_proj native closure')
                require(np.array_equal(boundary['latent'], np.concatenate((latent[:, 1:], po['predict:output'][:, -1:]), axis=1)) and
                    np.array_equal(boundary['history'], outgoing[:, 1:]) and
                    np.array_equal(boundary['latent'][0], after['latent_history']) and
                    np.array_equal(boundary['history'][0], after['action_history']), 'actual native endpoint identity closure')
                before = after


def validate_saved(p, r, events):
    require(len(r['rows']) == len(p['jobs']) == 16, 'full task/method/candidate denominator')
    require([e for e in events if e['kind'] == 'denominator'] == [dict(kind='denominator', rows=census(p['jobs']))], 'denominator sealed before execution')
    allowed = {'denominator', 'candidate_request', 'bank', 'candidate_return', 'model_before', 'model_after', 'sources_final',
        'resource_attempt', 'resource_return', 'io_input', 'io_return', 'program_packet', 'program_boundary'}
    require(all(e['kind'] in allowed for e in events), 'no failed/unknown/leaking events')
    for e in events:
        fields = ('kind', 'entry') if e['kind'].startswith('resource_') else {
            'denominator': ('kind', 'rows'), 'candidate_request': ('kind', 'job_sha256', 'row'),
            'bank': ('kind', 'slot', 'mode', 'bank'), 'candidate_return': ('kind', 'row'),
            'model_before': ('kind', 'hashes'), 'model_after': ('kind', 'hashes'),
            'sources_final': ('kind', 'sha256'),
            'io_input': ('kind', 'scope', 'phase', 'step', 'operation_index', 'arrays'),
            'io_return': ('kind', 'scope', 'phase', 'step', 'operation_index', 'arrays'),
            'program_packet': ('kind', 'scope', 'phase', 'step', 'operation_index', 'arrays'),
            'program_boundary': ('kind', 'scope', 'phase', 'step', 'operation_index', 'arrays')}[e['kind']]
        core.keys(e, fields)
    require([e for e in events if e['kind'] == 'candidate_request'] == [dict(kind='candidate_request',
        job_sha256=digest(j), row=dict(c, status='attempted')) for j, c in zip(p['jobs'], census(p['jobs']), strict=True)], 'all original requests predeclared without actions/outcomes')
    core.audit_ledger(r['resource_receipt'])
    require(r['resource_receipt']['caps'] == r['resource_receipt']['attempted'] == r['resource_receipt']['completed'] == p['profile']['resource_caps'], 'exact all-axis actual resources, not caps reported as actual')
    require([e['row'] for e in events if e['kind'] == 'candidate_return'] == r['rows'], 'durable full rows')
    banks = []
    for slot in (0, 1):
        for mode in ('fixed', 'adaptive'):
            job = next(j for j in p['jobs'] if j['slot'] == slot and j['mode'] == mode)
            require(job['program_type'] == 'progress', 'unsupported type cannot consume as progress')
            ledger = core.Ledger(p['profile']['resource_caps']); ledger.scope = dict(saved_array_recompute=True)
            banks.append(dict(kind='bank', slot=slot, mode=mode, bank=generate(p['tasks'][slot], job['response'], ledger)))
    require([e for e in events if e['kind'] == 'bank'] == banks, 'exact saved-J/ridge/rule/actions recomputation')
    winners = []
    for job, row in zip(p['jobs'], r['rows'], strict=True):
        require(all(row[k] == job[k] for k in ('slot', 'mode', 'index', 'horizon')) and row['status'] == 'model_replayed_not_admitted' and row['error'] is None, 'complete original task/candidate identity')
        bank = next(b['bank'] for b in banks if b['slot'] == row['slot'] and b['mode'] == row['mode'])
        require(row['actions'] == bank['actions'][row['index']] and row['whole'] == row['segmented'], 'sealed generated raw actions and double replay identity')
        t = p['tasks'][row['slot']]; terminal = core.checked_path(t, t['root_context'], row['actions'], row['whole'])
        residual = np.asarray(t['projection'])@(np.asarray(terminal['latent_history'][-1])-np.asarray(t['goal']))
        require(type(row['cost']) is float and row['cost'] == float(np.mean(residual**2)), 'original frozen Q terminal metric; no outcome')
    for slot in (0, 1):
        for mode in ('fixed', 'adaptive'):
            group = [row for row in r['rows'] if row['slot'] == slot and row['mode'] == mode]
            w = min(group, key=lambda row: (row['cost'], row['index']))
            winners.append(dict(slot=slot, mode=mode, index=w['index'], horizon=w['horizon'], cost=w['cost'], actions=w['actions'],
                terminal_context_sha256=digest(w['whole'][-1]), requested_depth=next(j['requested_depth'] for j in p['jobs'] if j['slot'] == slot and j['mode'] == mode),
                executed_extension_depth=1, physical_release=False))
    require(r['winners'] == winners, 'stable model-only selection; no physical release')
    validate_arrays(p, r['rows'], events, r['resource_receipt'])


def consume(output, expected):
    output = Path(output).absolute(); core.hash_value(expected)
    require(sha(output/'report.json') == expected, 'external actual report SHA')
    r = signal.read(output/'report.json'); p = verify(r['prepared'], r['prepare_sha256'])
    core.keys(r, ('schema', 'status', 'prepared', 'prepare_sha256', 'rows', 'winners', 'errors', 'resource_receipt',
        'model_before', 'model_after', 'source_manifest_sha256', 'sources_verified', 'saved_verification', 'host', 'wall_seconds',
        'CUDA_peak_bytes', 'journal_tail', 'artifact_sha256', *FLAGS))
    require(r['schema'] == NAME+'.report' and r['status'] == 'root_extensions_saved_not_admitted' and not r['errors'], 'complete engineering result only')
    require(all(r[k] is False for k in FLAGS) and r['sources_verified'] is True and r['source_manifest_sha256'] == digest(p['sources']), 'failclosed science and source seal')
    require(r['model_before'] == r['model_after'] and r['model_before']['original_state_dict'] == p['expected_model_hash'], 'model before/after frozen')
    require(r['saved_verification'] == dict(completed_upstream_passes=2, upstream_array_SVD_calls=8,
        partial_failed_pass_work_unknown=False, outside_native_generation_ledger=True), 'pre/post saved verification work separate from native generation ledger')
    require({f.name for f in output.iterdir()} == {'binding.json', 'report.json', 'events.jsonl'} and set(r['artifact_sha256']) == {'binding.json', 'events.jsonl'}, 'exact artifact set')
    require(signal.read(output/'binding.json') == dict(prepared=r['prepared'], prepare_sha256=r['prepare_sha256']), 'actual execution binding')
    for name, h in r['artifact_sha256'].items(): require(sha(output/name) == h, 'actual artifact hashes')
    events, tail = signal.journal(output/'events.jsonl'); require(tail == r['journal_tail'], 'journal chain/report binding')
    require([e['hashes'] for e in events if e['kind'] == 'model_before'] == [r['model_before']] and
        [e['hashes'] for e in events if e['kind'] == 'model_after'] == [r['model_after']] and
        [e['sha256'] for e in events if e['kind'] == 'sources_final'] == [digest(p['sources'])], 'actual final model/source journal binding')
    require(type(r['wall_seconds']) in (float, int) and 0 <= r['wall_seconds'] <= p['profile']['wall_seconds'], 'actual wall cap')
    require(type(r['CUDA_peak_bytes']) is list and len(r['CUDA_peak_bytes']) == 2 and all(type(v) is int and 0 <= v <= p['profile'][key]
        for v, key in zip(r['CUDA_peak_bytes'], ('cuda_allocated_bytes', 'cuda_reserved_bytes'))), 'actual CUDA memory cap')
    require(sum(f.stat().st_size for f in output.iterdir()) <= p['profile']['output_bytes'], 'actual total output cap')
    validate_saved(p, r, events)
    require(sha(output/'report.json') == expected and all(sha(f) == h for f, h in p['sources'].items()) and
        all(sha(output/n) == h for n, h in r['artifact_sha256'].items()), 'unchanged report/artifact/source closure after consume')
    per_branch = []
    for slot in (0, 1):
        for mode in ('fixed', 'adaptive'):
            local = [op for op in r['resource_receipt']['events'] if op['scope']['slot'] == slot and op['scope']['mode'] == mode]
            group = [row for row in r['rows'] if row['slot'] == slot and row['mode'] == mode]
            per_branch.append(dict(slot=slot, mode=mode, horizon=group[0]['horizon'], candidates=len(group),
                resources={axis: sum(op['delta'][axis] for op in local) for axis in core.AXES},
                inclusive_operation_wall_ns={name: sum(op['ended_ns']-op['started_ns'] for op in local if op['resource'] == name)
                    for name in sorted({op['resource'] for op in local})},
                distinct_action_buffers=len({digest(row['actions']) for row in group}),
                all_candidate_costs=[row['cost'] for row in group]))
    return dict(schema=NAME+'.consume', status='root_extensions_saved_verified_not_admitted', report_sha256=expected,
        denominator=16, winners=r['winners'], new_resources=r['resource_receipt']['completed'],
        per_branch=per_branch, historical_paid_resources=p['historical_paid_resources'], historical_acquisition=p['historical_acquisition'],
        saved_verification=dict(run=r['saved_verification'], this_consume_upstream_array_SVD_calls=4,
            this_consume_generator_array_SVD_calls=4, outside_native_generation_ledger=True),
        scientific_effect=None, complete_tree=False, child_adapter_missing=True, **FLAGS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--output', required=True)
    for cmd in ('verify', 'run'):
        a = sub.add_parser(cmd); a.add_argument('--prepared', required=True); a.add_argument('--sha256', required=True)
        if cmd == 'run': a.add_argument('--output', required=True); a.add_argument('--authorization', required=True)
    a = sub.add_parser('consume'); a.add_argument('--output', required=True); a.add_argument('--sha256', required=True)
    args = parser.parse_args(argv)
    if args.cmd == 'prepare': result = write_new(args.output, material())
    elif args.cmd == 'verify':
        p = verify(args.prepared, args.sha256); result = dict(status='verified_not_executed', denominator=len(p['jobs']), **FLAGS)
    elif args.cmd == 'run': result = run(args.prepared, args.sha256, args.output, args.authorization)
    else: result = consume(args.output, args.sha256)
    print(canonical(result).decode())


if __name__ == '__main__': main()
