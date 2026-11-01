"""MAIN-only actual-cp5 secant diagnostic: four groups, sixteen scored slots, no delivery."""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.spectral_reachability import agent_p3_goal_conditioned_R_actual_cp5_cuda_20261029_v1 as source
from experiments.spectral_reachability import agent_p3_reach_secant_kernel_20261029_v1 as kernel

base, tree, old, core, parent = source.base, source.tree, source.old, source.core, source.parent
signal, live = source.signal, source.live
require, canonical, digest, sha = core.require, core.canonical, core.digest, core.sha
ROOT = core.ROOT
NAME = 'agent_p3_reach_secant_actual_cp5_cuda_20261029_v1'
PROTOCOL = ROOT/'innerdocs/agent_p3_reach_secant_actual_cp5_protocol_20261029_v1.md'
PROTOCOL_SHA = '449873d6d60248cdeb34641842aa6a827eba4ab02dd0346e4ee570bc844cb37c'
AUTH = {
    'prepared': dict(path=str(ROOT/'results'/(source.NAME+'_prepare_MAIN01.json')),
        sha256='d54438930818af9bfd65c0f23a0666a19b320d3a62fb05256d5ac05beb798771'),
    'report': dict(path=str(ROOT/'results'/(source.NAME+'_cuda_MAIN01/report.json')),
        sha256='9bedec46c5c0e74b73c145935280b2efe400f617f8bbe7b72205fedb60a20d4f'),
    'receipt': dict(path=str(ROOT/'results'/(source.NAME+'_consume_MAIN01.json')),
        sha256='b50268a2018e6da1fbc46a4d19d701cd4a9a575cfaf581807cf29c01861d7ba8'),
}
PINS = {
    Path(source.__file__).resolve(): 'f6984725a676a79f4b4e9b32985ba6fda7957187ee4fd6f9bd1756d07afd353d',
    ROOT/'tests'/('test_'+source.NAME+'.py'): '094e5b68fe862652444a204cbf909210c0b82d718d6ebee6db83239a8c328c4d',
    ROOT/'innerdocs'/(source.NAME+'.md'): 'bd27ca6e956e6aaa843c30a04979b7ea0cff4df51f90b015d0143b8e5844f1a8',
    PROTOCOL: PROTOCOL_SHA,
}
FLAGS = dict(source.FLAGS, model_delivery_verified=False)
ROLE, roster = source.ROLE, source.roster
SETTLED = ('all_four_scored', 'surrogate_unavailable')


def surrogate_cost():
    return dict(tree.empty_axes(), program_compile_calls=1, array_SVD_calls=1, solve_calls=1)


def caps(h):
    base.signal_cost(h)
    return tree.sum_axes(surrogate_cost(), {k: 4*v for k, v in source.item_cost(h).items()})


def profile():
    return dict(source.profile(), resource_caps={str(h): caps(h) for h in (1, 3)})


def rules():
    return dict(protocol_sha256=PROTOCOL_SHA, root_context_role=ROLE, actual_cp5_start=True,
        radii=list(kernel.RADII), small_source_indices=[0, 1, 2, 3], groups=4, candidates_per_group=4,
        reuse_paid_zero=True, zero_in_new_cost=False, no_early_stop=True, no_direction_refit=True,
        exact_gate='saved I>source.zero.I AND saved R<source.zero.R; no epsilon',
        inherited_predict_API=196, inherited_jacobian_API=40, max_new_predict_API=80, max_new_jacobian_API=16,
        exact_gradient=False, scalar_backward_calls=None, total_FLOPs=None, **FLAGS)


def source_material(p, r, receipt, rebuilt):
    """Pure extraction AFTER an entire upstream consume; names alone confer no authority."""
    require(canonical(receipt) == canonical(rebuilt), 'entire persisted source receipt equals actual frozen consume')
    for doc, suffix in ((p, 'prepare'), (r, 'report'), (receipt, 'consume')):
        require(doc['schema'] == source.NAME+'.'+suffix and all(doc[k] is False for k in source.FLAGS), 'real upstream schema/full_C3_demonstrated flags')
    require(r['status'] == 'four_slots_settled_not_admitted' and r['errors'] == []
        and receipt['status'] == 'actual_cp5_goal_R_model_deliveries_verified_NOT_physical'
        and receipt['denominator'] == 4 and receipt['candidate_denominator'] == 32 and receipt['controls'] == 4,
        'whole completed source cohort')
    require(r['prepared'] == receipt['prepared'] == AUTH['prepared']['path']
        and r['prepare_sha256'] == receipt['prepare_sha256'] == AUTH['prepared']['sha256']
        and receipt['report_sha256'] == AUTH['report']['sha256'], 'exact source three-way authority join')
    require(r['sources_verified'] is True and r['source_manifest_sha256'] == digest(p['sources'])
        and r['model_before'] == r['model_after'] and r['model_before']['original_state_dict'] == p['expected_model_hash'], 'source closure/model unchanged')
    require(r['classes_before'] == r['classes_after'] and r['classes_before'], 'source installed class closure')
    for c in r['classes_before']: require(p['sources'].get(c['path']) == c['sha256'], 'source class exact pin')
    require(p['roster'] == roster() and len(p['tasks']) == len(p['provenance']) == 2
        and len(r['rows']) == len(receipt['arms']) == 4, 'all four source groups/two tasks')
    groups = []
    for identity, row, arm in zip(roster(), r['rows'], receipt['arms'], strict=True):
        require(all(type(row[k]) is int and row[k] == v and arm[k] == v for k, v in identity.items()), 'fixed source slot/root/horizon')
        root, h = row['root'], row['horizon']; task, pro = source.input_pair(p['tasks'][root], p['provenance'][root])
        result = row['result']
        require(result['schema'] == source.NAME+'.group' and result['backend_kind'] == 'HOST_CUDA'
            and result['task_sha256'] == digest(task) and result['provenance_sha256'] == digest(pro)
            and result['source_closure_sha256'] == digest(p['sources']) and result['root_context_role'] == ROLE
            and result['actual_cp5_start'] is True and result['errors'] == []
            and all(result[k] is False and arm[k] is False for k in source.FLAGS), 'whole source result identities')
        require(row['status'] == result['status'] == arm['status'] == 'no_witness' and arm['complete'] is True
            and result['delivery'] is None and arm['delivery'] is None and arm['no_witness'] is True,
            'pinned source no-witness diagnostic, not a physical plan')
        require(len(result['R_groups']) == 1 and arm['resources'] == result['resource_receipt'], 'source full group/receipt resources')
        rr = arm['resources']; attempted, completed = core.counters(rr['attempted']), core.counters(rr['completed'])
        require(attempted == completed, 'all inherited source attempts returned')
        g = result['R_groups'][0]; bank = g['bank']
        require([x['index'] for x in g['rows']] == list(range(8)) and len(bank['candidates']) == 8
            and g['control']['index'] == 'zero' and g['control']['status'] == 'complete', 'all original zero+8')
        _, local, actual_h, native = tree.spectral._mapping(task, [], [], bank['mapping'])
        require(actual_h == h and bank['local_task'] == local and bank['native_response'] == native
            and bank['provenance']['task_sha256'] == digest(local)
            and bank['provenance']['signal_sha256'] == digest(native)
            and bank['source_capsule_bytes_verified_here'] is True
            and bank['kernel_source_artifact_authenticated_here'] is False,
            'authenticated native LOCAL task/signal, not a same-name root alias')
        require(bank['kernel_flags'] == source.numeric.FLAGS, 'preserve numerical kernel C3 ABI separately')
        small, old_items = [], []
        for i, (candidate, item) in enumerate(zip(bank['candidates'], g['rows'], strict=True)):
            require(candidate['index'] == item['index'] == i, 'candidate exact order')
            if candidate['actions'] is None:
                require(item['status'] == 'blocked_degenerate_direction' and item['actions'] is None
                    and item['eligible'] is None, 'source null, not scored zero')
                immediate = reach = None
            else:
                require(item['status'] == 'complete' and item['actions'] == candidate['actions'], 'raw action source pairing')
                immediate, reach = kernel.score(item['immediate']), kernel.score(item['reach'])
                require(item['eligible'] is (immediate > g['control']['immediate'] and reach < g['control']['reach']), 'unchanged source exact gate')
            old_items.append(dict(index=i, status=item['status'], immediate=immediate, reach=reach,
                eligible=item['eligible'], row_sha256=digest(item), actions_sha256=digest(item['actions'])))
            if i < 4:
                small.append(dict(index=i, direction=candidate['direction'], sign=candidate['sign'], radius=candidate['radius'],
                    params=deepcopy(candidate['params']), actions=deepcopy(candidate['actions']),
                    status=item['status'], immediate=immediate, reach=reach))
        kernel.validate(h, bank['directions'], small)
        zero = dict(index='zero', immediate=kernel.score(g['control']['immediate']), reach=kernel.score(g['control']['reach']),
                    row_sha256=digest(g['control']), actions_sha256=digest(g['control']['actions']))
        require(np.asarray(g['control']['actions']).shape == (5*h, 2)
            and np.all(np.asarray(g['control']['actions']) == 0), 'original explicitly paid zero actions')
        valid = sum(c['actions'] is not None for c in bank['candidates'])
        require(attempted == source.expected_usage(h, valid, False), 'exact inherited whole source costs')
        groups.append(dict(**identity, source_result_sha256=digest(result), source_bank_sha256=digest(bank),
            source_group_sha256=digest(g), source_manifest_sha256=digest(p['sources']),
            task_sha256=digest(task), provenance_sha256=digest(pro), local_task_sha256=digest(local),
            native_response_sha256=digest(native), original_items=[zero]+old_items, zero=zero,
            directions=deepcopy(bank['directions']), small_rows=small,
            inherited_resources=dict(attempted=attempted, completed=completed, receipt_sha256=digest(rr)),
            report_origin_pointer='/rows/'+str(identity['slot'])+'/result/R_groups/0'))
    inherited = tree.sum_axes(*[g['inherited_resources']['attempted'] for g in groups])
    require(inherited['predict_API'] == 196 and inherited['jacobian_API'] == 40, 'pinned cumulative inherited 196/40; never fill missing with zero')
    return groups, inherited


def read_pin(ref, cap):
    p = old.regular(ref['path'], cap)
    require(sha(p) == ref['sha256'], 'exact saved authority bytes')
    out = signal.read(p, cap); require(sha(p) == ref['sha256'], 'unchanged authority read')
    return out


def load_source():
    """MAIN production-only: do NOT call this in worker RAM tests."""
    p = read_pin(AUTH['prepared'], 16*1024**2)
    r = read_pin(AUTH['report'], source.profile()['final_reserve_bytes'])
    receipt = read_pin(AUTH['receipt'], 128*1024**2)
    rebuilt = source.consume(Path(AUTH['report']['path']).parent, AUTH['report']['sha256'],
                             AUTH['prepared']['path'], AUTH['prepared']['sha256'])
    groups, inherited = source_material(p, r, receipt, rebuilt)
    require(all(sha(v['path']) == v['sha256'] for v in AUTH.values()), 'unchanged full source authorities')
    return p, groups, inherited


def own_files():
    return [Path(__file__).resolve(), Path(kernel.__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')]


def build():
    require(all(sha(p) == h for p, h in PINS.items()), 'frozen approved protocol and producer triple')
    p, groups, inherited = load_source()
    sources = dict(p['sources']); sources.update({str(p.resolve()): sha(p) for p in [*PINS, *own_files()]})
    sources.update({v['path']: v['sha256'] for v in AUTH.values()})
    return dict(schema=NAME+'.prepare', tasks=p['tasks'], provenance=p['provenance'], source_groups=groups,
        authorities=deepcopy(AUTH), inherited_resources=inherited, cp5_tasks=p['cp5_tasks'], policy=p['policy'],
        expected_model_hash=p['expected_model_hash'], source_manifest_sha256=digest(p['sources']),
        sources=sources, roster=roster(), profile=profile(), precision=old.precision_contract(), rules=rules(), **FLAGS)


def group(task, pro, h, inherited, factory, sink, *, source_sha, permit_ram=False, guard=lambda: None,
          saved_calls=None, clock=time.monotonic_ns):
    task, pro = source.input_pair(task, pro); base.signal_cost(h)
    require(inherited['horizon'] == h and inherited['task_sha256'] == digest(task)
        and inherited['provenance_sha256'] == digest(pro), 'same complete observed task/source identity')
    kind = 'RAM_FIXTURE' if permit_ram else 'HOST_CUDA'
    ledger = core.Ledger(caps(h), clock=clock)
    session = tree.Session(task, factory, sink, ledger, kind, guard, None, saved_calls); ledger.sink = session.event
    r = dict(schema=NAME+'.group', status='not_started', backend_kind=kind, horizon=h,
        task_sha256=digest(task), provenance_sha256=digest(pro), source_closure_sha256=source_sha,
        source_group_sha256=digest(inherited), root_context_role=ROLE, actual_cp5_start=True,
        surrogate=None, rows=[dict(index=i, radius=radius, status='not_requested', reason=None, actions=None,
                                   eligible=None) for i, radius in enumerate(kernel.RADII)],
        eligible_indices=[], errors=[], delivery=None, **FLAGS)
    try:
        guard(); ledger.scope = dict(phase='secant', horizon=h, parent_context_sha256=digest(task['root_context']))
        require(all(ledger.attempted[k]+caps(h)[k] <= ledger.caps[k] for k in core.AXES), 'whole surrogate+four-item atomic cap')
        session.event(dict(kind='secant_attempt', source_group_sha256=digest(inherited)))
        bank = kernel.build(h, inherited['directions'], inherited['small_rows'], ledger.call,
                            on_start=lambda value: r.update(surrogate=value))
        session.event(dict(kind='secant_return', surrogate=deepcopy(bank)))
        if bank['status'] == 'surrogate_unavailable':
            for row in r['rows']:
                row.update(status='blocked_surrogate', reason=bank['reason'])
                session.event(dict(kind='secant_item_blocked', row=deepcopy(row)))
            r['status'] = 'surrogate_unavailable'
        else:
            require(bank['status'] == 'complete', 'complete secant kernel')
            local = parent.parent_task(task, task['root_context'])
            for row, candidate in zip(r['rows'], bank['candidates'], strict=True):
                guard(); raw = candidate['actions']; row.update(status='attempted', actions=deepcopy(raw), params=deepcopy(candidate['params']))
                ledger.scope = dict(phase='secant_item', index=row['index'], horizon=h,
                                    parent_context_sha256=digest(task['root_context']))
                session.event(dict(kind='secant_item_attempt', index=row['index'], candidate=deepcopy(candidate)))
                path = ledger.call('program_compile', lambda: ledger.call('replay', lambda: session.replay(local, raw)))
                post = core.checked_path(task, task['root_context'], raw, path)
                row.update(path=path, postcontext=post, immediate=ledger.call('metric', lambda: core.model_cost(task, post)))
                posttask = parent.parent_task(task, post)
                response = ledger.call('signal_compile', lambda: session.signal(posttask, 1))
                nominal, pointer = tree.nominal_path(session.records[-1], posttask, response)
                residual = np.asarray(task['goal'], np.float64)-np.asarray(response['nominal'], np.float64)
                solved = ledger.call('solve', lambda: ledger.call('array_svd', lambda: tree.kernel.original.bounded_ridge(response['jacobian'], residual)))
                suffix = tree.components.suffix_raw(solved['params'])
                suffix_path = ledger.call('replay', lambda: session.replay(posttask, suffix))
                terminal = core.checked_path(task, post, suffix, suffix_path)
                row.update(status='complete', signal=response, nominal_path=nominal, nominal_pointer=pointer,
                    solve=solved, reach=solved['own_reach_value'], suffix_actions=suffix, suffix_path=suffix_path,
                    bounded_suffix_cost=ledger.call('metric', lambda: core.model_cost(task, terminal)))
                row['nongreedy'] = row['immediate'] > inherited['zero']['immediate']
                row['better_reach'] = row['reach'] < inherited['zero']['reach']
                row['eligible'] = row['nongreedy'] and row['better_reach']
                row['rejection_factors'] = ([] if row['nongreedy'] else ['immediate_not_above_zero'])+([] if row['better_reach'] else ['reach_not_below_zero'])
                session.event(dict(kind='secant_item_return', row=deepcopy(row)))
            r['eligible_indices'] = [row['index'] for row in r['rows'] if row['eligible']]
            require(ledger.attempted == ledger.completed == caps(h), 'four items all paid; no early stop at eligibility')
            r['status'] = 'all_four_scored'
        guard()
    except Exception as exc:
        r.update(status='partial_runtime_failure', errors=[dict(type=type(exc).__name__, message=str(exc))])
        for row in r['rows']:
            if row['status'] == 'attempted': row['status'] = 'failed'
            elif row['status'] == 'not_requested': row.update(status='blocked_after_runtime_failure', reason='prior_failure')
    r.update(native_calls=deepcopy(session.records), resource_receipt=ledger.receipt())
    core.audit_ledger(r['resource_receipt'])
    return r


def execute(p, factory, store, *, permit_ram=False, check=lambda: None):
    rows = [dict(**x, status='not_requested', result=None) for x in roster()]; failed = False
    store.event(dict(kind='denominator', rows=deepcopy(rows), candidates=16))
    for row in rows:
        if failed:
            row.update(status='blocked_after_runtime_failure'); continue
        slot = row['slot']; begin = getattr(store, 'bytes', 0); start = time.monotonic()
        store.event(dict(kind='group_attempt', identity=roster()[slot]))
        sink = type('Sink', (), {'event': lambda self, e: store.event(dict(kind='group_event', slot=slot, payload=deepcopy(e)))})()
        def guard():
            check(); require(time.monotonic()-start <= p['profile']['group_wall_seconds'], 'group wall cap')
        try:
            guard()
            result = group(p['tasks'][row['root']], p['provenance'][row['root']], row['horizon'], p['source_groups'][slot],
                factory, sink, source_sha=digest(p['sources']), permit_ram=permit_ram, guard=guard)
            row.update(status=result['status'], result=result, wall_seconds=time.monotonic()-start); failed = bool(result['errors'])
            store.event(dict(kind='group_return', row=deepcopy(row)))
            require(getattr(store, 'bytes', 0)-begin+len(canonical(row)) <= p['profile']['group_output_bytes'], 'full group byte cap')
            guard()
        except Exception as exc:
            row.update(status='outer_runtime_failure', error=dict(type=type(exc).__name__, message=str(exc))); failed = True
            store.event(dict(kind='group_outer_failure', row=deepcopy(row)))
    return rows


def accounting(p, rows):
    inherited = core.counters(p['inherited_resources'])
    attempted = tree.sum_axes(*[core.counters(x['result']['resource_receipt']['attempted']) for x in rows if x['result'] is not None])
    completed = tree.sum_axes(*[core.counters(x['result']['resource_receipt']['completed']) for x in rows if x['result'] is not None])
    maximum = tree.sum_axes(*[caps(x['horizon']) for x in roster()])
    return dict(inherited=deepcopy(inherited), new_attempted=attempted, new_completed=completed, new_caps=maximum,
        cumulative_attempted=tree.sum_axes(inherited, attempted), cumulative_completed=tree.sum_axes(inherited, completed),
        cumulative_caps=tree.sum_axes(inherited, maximum), scalar_backward_calls=None, total_FLOPs=None,
        inherited_AD_primal_predict_API=44, inherited_Jacobian_elements=84480,
        new_AD_primal_predict_API=attempted['jacobian_API'], new_Jacobian_elements=1920*attempted['jacobian_API'])


def exclusive(path):
    p = Path(path).absolute()
    require(p.parent == ROOT/'results' and p.name.startswith(NAME) and p.resolve() == p
            and not p.exists() and not p.is_symlink(), 'new exclusive own output')
    return p


def write_new(path, value, cap=16*1024**2):
    p = exclusive(path); raw = canonical(value)
    require(len(raw) <= cap and shutil.disk_usage(p.parent).free >= profile()['disk_floor_bytes']+len(raw), 'metadata cap/floor')
    with p.open('xb') as f:
        require(f.write(raw) == len(raw), 'complete metadata'); f.flush(); os.fsync(f.fileno())
    signal.fsync_dir(p.parent)
    return dict(path=str(p), sha256=sha(p))


def prepare(output): return write_new(output, build())


def verify(prepared, expected):
    core.hash_value(expected); path = old.regular(prepared, 16*1024**2)
    require(sha(path) == expected, 'actual own prepare SHA'); p = signal.read(path, 16*1024**2)
    require(p == build() and sha(path) == expected, 'whole source receipt/task/budget/source reconstitution')
    return p


def run(prepared, expected, output, *, authorization):
    require(authorization == 'MAIN_APPROVED_HOST_CUDA', 'MAIN source audit and HOST CUDA authorization')
    started = time.monotonic(); p = verify(prepared, expected); output = exclusive(output)
    require(shutil.disk_usage(output.parent).free >= p['profile']['output_bytes']+p['profile']['disk_floor_bytes'], '2GiB output allowance plus floor')
    store = base.Store(output, p['profile']); store.started = started
    rows = [dict(**x, status='blocked_before_model', result=None) for x in roster()]
    model = None; errors = []; before = after = classes = classes_after = host = None; guards = []; source_ok = False
    try:
        store.put('binding.json', dict(prepared=str(Path(prepared).absolute()), prepare_sha256=expected))
        store.event(dict(kind='source_before', sha256=digest(p['sources'])))
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
        classes = old.model_sources(model, p); store.event(dict(kind='model_before', hashes=before, classes=classes))
        forwards = [(m, 'forward' in m.__dict__, m.__dict__.get('forward')) for m in model.modules()]
        with torch.autocast(device_type='cuda', enabled=False), torch.autocast(device_type='cpu', enabled=False), sdpa_kernel(SDPBackend.MATH):
            def guard(): store.check(); old.checked_precision(torch)
            guards.append(old.checked_precision(torch)); store.event(dict(kind='precision_before', value=guards[-1]))
            rows = execute(p, lambda t, s: source.prior.NativeDelivery(model, device, t, s), store, check=guard)
            guards.append(old.checked_precision(torch)); store.event(dict(kind='precision_after', value=guards[-1]))
        torch.cuda.synchronize(device)
        require(all(('forward' in m.__dict__) == had and m.__dict__.get('forward') is v for m, had, v in forwards), 'instance hooks restored')
    except BaseException as exc:
        errors.append(dict(stage='run', type=type(exc).__name__, message=str(exc)))
    finally:
        if model is not None:
            try:
                after = signal.model_hash(model); classes_after = old.model_sources(model, p)
                require(before == after and classes == classes_after and all(not m.training for m in model.modules())
                    and all(not x.requires_grad for x in model.parameters()), 'frozen model/classes beforeafter')
                store.event(dict(kind='model_after', hashes=after, classes=classes_after))
            except BaseException as exc: errors.append(dict(stage='model_final', message=str(exc)))
        try:
            verify(prepared, expected); store.check(); source_ok = True
            store.event(dict(kind='source_after', sha256=digest(p['sources'])))
        except BaseException as exc: errors.append(dict(stage='source_final', message=str(exc)))
        store.close(); settled = all(x['status'] in SETTLED for x in rows)
        report = dict(schema=NAME+'.report', status='four_groups_settled_NOT_delivery' if settled and not errors else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()), prepare_sha256=expected, rows=rows, errors=errors,
            model_before=before, model_after=after, classes_before=classes, classes_after=classes_after,
            host=host, precision=guards, sources_verified=source_ok, source_manifest_sha256=digest(p['sources']),
            wall_seconds=store.clock()-store.started, CUDA_peak_bytes=store.memory() if store.memory else None,
            journal_tail=store.tail, artifact_sha256={n: sha(output/n) for n in ('binding.json', 'events.jsonl') if (output/n).exists()},
            accounting=accounting(p, rows), delivery=None, **FLAGS)
        store.put('report.json', report)
    require(report['status'] == 'four_groups_settled_NOT_delivery', 'partial saved; no retry '+repr(errors))
    return dict(output=str(output), report_sha256=sha(output/'report.json'))


def validate_group(task, pro, h, inherited, r, events, *, source_sha, permit_fixture=False):
    require(r['backend_kind'] == 'HOST_CUDA' or permit_fixture and r['backend_kind'] == 'RAM_FIXTURE', 'no RAM production')
    require(all(e['kind'] == 'io' for e in events), 'own native IO envelope')
    allowed = old.RESOURCE | old.CALL_ARRAYS | {'direction', 'native_call_attempt', 'native_call_return', 'native_call_failure',
        'secant_attempt', 'secant_return', 'secant_item_blocked', 'secant_item_attempt', 'secant_item_return'}
    require(all(e['payload']['kind'] in allowed for e in events), 'unknown secant event')
    require(r['resource_receipt']['caps'] == caps(h), 'exact standalone new cost cap')
    payloads = [e['payload'] for e in events]; old.resource_census(r['resource_receipt'], payloads)
    calls = r['native_calls']; require([x['call_id'] for x in calls] == list(range(len(calls))), 'native invocation census')
    require(all(e['call'] is None or type(e['call']) is int and 0 <= e['call'] < len(calls) for e in events), 'no orphan call')
    for c in calls:
        require([e['payload'] for e in events if e['call'] == c['call_id']] == c['transcript'], 'complete native call transcript')
        old.audit_native(c['transcript'])
    replay_events = []; sink = type('Sink', (), {'event': lambda self, e: replay_events.append(deepcopy(e))})()
    replay = group(task, pro, h, inherited, None, sink, source_sha=source_sha, permit_ram=permit_fixture,
                   saved_calls=calls, clock=tree.replay_clock(r['resource_receipt'], payloads))
    require(canonical(replay) == canonical(r), 'whole saved secant/scores/null/gate/resource reconstruction')
    require([e for e in replay_events if e['call'] is None] == [e for e in events if e['call'] is None], 'all noncall events exact order/census')
    complete = r['status'] in SETTLED; scored = sum(x['status'] == 'complete' for x in r['rows'])
    if complete:
        require(len(r['rows']) == 4 and len(calls) == 3*scored and scored in (0, 4), 'all four scored or explicit four null; no early stop')
        require(r['resource_receipt']['attempted'] == r['resource_receipt']['completed'], 'all actual attempts returned')
    return dict(horizon=h, status=r['status'], complete=complete, scored=scored, eligible_indices=r['eligible_indices'],
        resources=r['resource_receipt'], new_AD_primal_predict_API=r['resource_receipt']['attempted']['jacobian_API'],
        new_Jacobian_elements=1920*r['resource_receipt']['attempted']['jacobian_API'], delivery=None, **FLAGS)


def validate_rows(p, rows, events, *, permit_fixture=False):
    require(len(rows) == 4 and [(r['slot'], r['root'], r['horizon']) for r in rows] ==
            [(r['slot'], r['root'], r['horizon']) for r in roster()], 'fixed four groups/sixteen slots')
    top = {'denominator', 'group_attempt', 'group_return', 'group_outer_failure', 'source_before', 'source_after',
           'model_before', 'model_after', 'precision_before', 'precision_after'}
    require(all(e['kind'] in top | {'group_event'} for e in events), 'unknown top event')
    require(all(type(e['slot']) is int and 0 <= e['slot'] < 4 for e in events if e['kind'] == 'group_event'), 'orphan slot')
    require([e for e in events if e['kind'] == 'denominator'] == [dict(kind='denominator', candidates=16,
        rows=[dict(**x, status='not_requested', result=None) for x in roster()])], 'initial full denominator')
    failed = False; position = -1; out = []
    for row, identity in zip(rows, roster(), strict=True):
        es = [e['payload'] for e in events if e['kind'] == 'group_event' and e['slot'] == row['slot']]
        starts = [i for i, e in enumerate(events) if e['kind'] == 'group_attempt' and e['identity'] == identity]
        ends = [i for i, e in enumerate(events) if e['kind'] == 'group_return' and e['row'] == row]
        if row['result'] is None:
            require(failed and row['status'] == 'blocked_after_runtime_failure' and not starts and not ends and not es, 'failure-only subsequent stop'); continue
        require(not failed and row['status'] == row['result']['status'] and len(starts) == len(ends) == 1
                and position < starts[0] < ends[0], 'ordered full dispatch')
        require(all(starts[0] < i < ends[0] for i, e in enumerate(events) if e['kind'] == 'group_event' and e['slot'] == row['slot']), 'events within group')
        position = ends[0]
        require(type(row['wall_seconds']) in (int, float) and 0 <= row['wall_seconds'] <= p['profile']['group_wall_seconds'], 'group wall cap')
        summary = validate_group(p['tasks'][row['root']], p['provenance'][row['root']], row['horizon'], p['source_groups'][row['slot']],
            row['result'], es, source_sha=digest(p['sources']), permit_fixture=permit_fixture)
        out.append(dict(slot=row['slot'], root=row['root'], **summary)); failed = bool(row['result']['errors'])
    require(len([e for e in events if e['kind'] == 'group_attempt']) == len(out) == len([e for e in events if e['kind'] == 'group_return']), 'no hidden dispatch')
    require(not any(e['kind'] == 'group_outer_failure' for e in events), 'outer failure retained not certified')
    return out


def consume(output, report_sha, prepared, prepare_sha):
    output = Path(output).absolute(); require(output.resolve() == output and not output.is_symlink(), 'exact own output')
    core.hash_value(report_sha); rp = old.regular(output/'report.json', profile()['final_reserve_bytes'])
    require(sha(rp) == report_sha, 'actual report SHA'); r = signal.read(rp, profile()['final_reserve_bytes']); p = verify(prepared, prepare_sha)
    require(r['schema'] == NAME+'.report' and r['prepared'] == str(Path(prepared).absolute()) and r['prepare_sha256'] == prepare_sha, 'own authority schema')
    require(not r['errors'] and all(r[k] is False for k in FLAGS) and r['delivery'] is None
        and r['sources_verified'] is True and r['source_manifest_sha256'] == digest(p['sources']), 'science/source/no-delivery closure full_C3_demonstrated')
    require(r['model_before'] == r['model_after'] and r['model_before']['original_state_dict'] == p['expected_model_hash'], 'same inherited frozen model')
    require(r['classes_before'] == r['classes_after'] and r['classes_before'], 'installed class closure')
    for c in r['classes_before']: require(p['sources'].get(c['path']) == c['sha256'], 'installed class exact pin')
    require(r['precision'] == [old.precision_contract()]*2 and r['host']['device'] == 'cuda:0'
        and r['host']['allocation_sum'] == 4. and r['host']['visible_nvidia_nodes'], 'HOST CUDA and full precision guards')
    require(set(r['artifact_sha256']) == {'binding.json', 'events.jsonl'}
        and {x.name for x in output.iterdir()} == {'binding.json', 'events.jsonl', 'report.json'}, 'exact file census; no hidden capsules')
    for n, h in r['artifact_sha256'].items(): require(sha(old.regular(output/n, profile()['output_bytes'])) == h, 'all artifact bytes')
    require(signal.read(output/'binding.json') == dict(prepared=r['prepared'], prepare_sha256=prepare_sha), 'persisted binding')
    sizes = {}; es = base.journal(output, r, sizes)
    for phase in ('before', 'after'):
        require([e for e in es if e['kind'] == 'model_'+phase] == [dict(kind='model_'+phase, hashes=r['model_'+phase], classes=r['classes_'+phase])], 'model journal')
        require([e for e in es if e['kind'] == 'source_'+phase] == [dict(kind='source_'+phase, sha256=digest(p['sources']))], 'source journal')
        require([e['value'] for e in es if e['kind'] == 'precision_'+phase] == [old.precision_contract()], 'precision journal')
    summaries = validate_rows(p, r['rows'], es)
    require(r['status'] == 'four_groups_settled_NOT_delivery' and len(summaries) == 4 and all(x['complete'] for x in summaries), 'all four settled including null')
    require(r['accounting'] == accounting(p, r['rows']), 'entire inherited/new/cumulative resource vector')
    require(0 <= r['wall_seconds'] <= p['profile']['wall_seconds'] and sum(x.stat().st_size for x in output.iterdir()) <= p['profile']['output_bytes'], 'wall/output caps')
    require(type(r['CUDA_peak_bytes']) is list and len(r['CUDA_peak_bytes']) == 2 and all(type(v) is int and 0 <= v <= p['profile'][k]
        for v, k in zip(r['CUDA_peak_bytes'], ('cuda_allocated_bytes', 'cuda_reserved_bytes'))), 'CUDA memory caps')
    require(all(sizes[row['slot']]+len(canonical(row)) <= p['profile']['group_output_bytes'] for row in r['rows']), 'full group byte caps')
    require(sha(rp) == report_sha and all(sha(f) == h for f, h in p['sources'].items()), 'unchanged full sources and report')
    return dict(schema=NAME+'.consume', status='all4_actual_cp5_secant_groups_verified_NOT_delivery',
        prepared=r['prepared'], prepare_sha256=prepare_sha, report_sha256=report_sha, authorities=p['authorities'],
        protocol_sha256=PROTOCOL_SHA, groups=4, candidate_denominator=16, inherited_candidates=32, inherited_controls=4,
        arms=summaries, accounting=r['accounting'], delivery=None, **FLAGS)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('prepare'); q.add_argument('--output', required=True)
    for cmd in ('verify', 'run', 'consume'):
        q = sub.add_parser(cmd); q.add_argument('--prepared', required=True); q.add_argument('--prepare-sha', required=True)
        if cmd in ('run', 'consume'): q.add_argument('--output', required=True)
        if cmd == 'run': q.add_argument('--authorization', required=True)
        if cmd == 'consume': q.add_argument('--report-sha', required=True); q.add_argument('--receipt', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'prepare': out = prepare(a.output)
    elif a.cmd == 'verify': out = dict(verified=True, schema=verify(a.prepared, a.prepare_sha)['schema'])
    elif a.cmd == 'run': out = run(a.prepared, a.prepare_sha, a.output, authorization=a.authorization)
    else: out = write_new(a.receipt, consume(a.output, a.report_sha, a.prepared, a.prepare_sha), 128*1024**2)
    print(canonical(out).decode())


if __name__ == '__main__': main()
