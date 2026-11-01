"""MAIN-only goal-conditioned R8 at actual cp5; no physical release or fallback."""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.spectral_reachability import agent_p3_actual_cp5_r_delivery_20261029_v1 as prior
from experiments.spectral_reachability import agent_p3_goal_conditioned_R_bank_20261029_v1 as numeric

base, tree, old, core, parent = prior.base, prior.tree, prior.old, prior.core, prior.parent
signal, live = prior.signal, prior.live
require, canonical, digest, sha = core.require, core.canonical, core.digest, core.sha
ROOT = core.ROOT
NAME = 'agent_p3_goal_conditioned_R_actual_cp5_cuda_20261029_v1'
PROTOCOL = ROOT/'innerdocs/main_p3_goal_conditioned_R_actual_cp5_protocol_20261029_v1.md'
PROTOCOL_SHA = '24012da9b83f80491bb8fd6c9ac158d01d6752eb3f4254fd32902da44cbd84f0'
PINS = {
    Path(numeric.__file__).resolve(): '367cdba6fd97888504b5005b1bd72512c111e522e783aaaca164671e0625b6c8',
    ROOT/'tests'/('test_'+numeric.NAME+'.py'): '97e9e0238d5f9a69fbdcc479a5eec9399d05f113962344681d4c3460b192654f',
    ROOT/'innerdocs'/(numeric.NAME+'.md'): '575fed6d31d0ea363bbbb06074ca93b6e3ba4ca8ade01461cd098a5ab79b75e9',
    Path(prior.__file__).resolve(): '09af91981549547c5f770ad3d0ffa42ad92b48e37daefd612191d18ec9755a26',
    ROOT/'tests'/('test_'+prior.NAME+'.py'): '9cd129b701d5f7d87cfaa74a10ea8c709bf140bbc12687582d087537ab93c70f',
    ROOT/'innerdocs'/(prior.NAME+'.md'): '79dc91fc2f1304d9d6c526614f99fba900c7b61ad05e579c280d04a4df9e31d1',
    PROTOCOL: PROTOCOL_SHA,
}
FLAGS = dict(prior.FLAGS)  # Production ABI uses full_C3_demonstrated, NOT the numerical kernel's C3.
ROLE, SETTLED = prior.ROLE, prior.SETTLED
roster, input_pair = prior.roster, prior.input_pair


def caps(h):
    out = prior.caps(h); out['solve_calls'] += 1
    return out


def bank_cost():
    return dict(tree.empty_axes(), **numeric.COST)


def item_cost(h):
    v = tree.kernel.minimum(h)
    require(all(v[k] % 9 == 0 for k in core.AXES), 'nine equal original evaluator tariffs')
    return {k: v[k]//9 for k in core.AXES}


def group_cost(h, valid=8):
    core.integer(valid); require(valid <= 8, 'fixed eight slots')
    return tree.sum_axes(bank_cost(), {k: (valid+1)*v for k, v in item_cost(h).items()})


def expected_usage(h, valid, delivery):
    return tree.sum_axes(base.signal_cost(h), group_cost(h, valid),
                         prior.verification_cost(h) if delivery else tree.empty_axes())


def profile():
    return dict(prior.profile(), wall_seconds=3600, group_wall_seconds=1800,
                final_reserve_bytes=512*1024**2, resource_caps={str(h): caps(h) for h in (1, 3)})


def rules():
    return dict(protocol_sha256=PROTOCOL_SHA, cp5_tasks_sha256=prior.CP5_SHA,
        root_context_role=ROLE, actual_cp5_start=True, bank=numeric.contract(),
        numerical_integration='frozen validate_inputs/direction_data/slots and literal generate equations; original shared ledger',
        all8_preserved=True, blocked_slots='null, no calls, no substitution, no padding',
        separate_zero_evaluation_paid=True, selection=['immediate', 'candidate_index'],
        segmented_raw_steps={'1': [5], '3': [5, 10]}, commit_raw_steps=5,
        suffix_is_plan=False, scalar_backward_calls=None, total_FLOPs=None,
        failure='preserve partial and all four slots; block later dispatch; no retry',
        exploratory_known_roots=True, blind_confirmation=False, **FLAGS)


def make_bank(root, mapping, session, *, on_start):
    """Bind actual local signal IO; reuse frozen math with per-operation durable original metering.

    Frozen build_bank owns a separate non-durable ledger and has no callback
    injection. Its literal generate equations are integrated here, not patched;
    validate_inputs/direction_data/slots/contract remain the frozen functions.
    """
    ledger = session.ledger
    _, task, h, response = tree.spectral._mapping(root, [], [], mapping)
    evidence = tree.source_evidence(session, mapping)
    source = tree.spectral._evidence(task, h, response, mapping, evidence, session.kind == 'RAM_FIXTURE')
    authenticated = False
    if session.kind == 'HOST_CUDA':
        path = old.regular(evidence['artifact_path'], 64*1024**2)
        require(sha(path) == evidence['artifact_sha256'] == digest(evidence['native_call'])
            and signal.read(path, 64*1024**2) == evidence['native_call'], 'persisted source capsule full bytes/native IO')
        authenticated = True
    provenance = dict(kind='SAVED_REFERENCE' if authenticated else 'RAM_FIXTURE',
        artifact_path=evidence['artifact_path'], artifact_sha256=evidence['artifact_sha256'],
        call_id=evidence['call_id'], origin_index=evidence['origin_index'], nominal_index=evidence['nominal_index'],
        jacobian_index=evidence['jacobian_index'], task_sha256=digest(task), signal_sha256=digest(response),
        paid_source_resources=deepcopy(source['resources']))
    q, goal, nominal, j, h = numeric.validate_inputs(task, response, provenance)
    out = dict(schema=NAME+'.bank', status='running', rule=numeric.contract(),
        local_task=deepcopy(task), native_response=deepcopy(response), mapping=deepcopy(mapping),
        provenance=provenance, evidence=evidence, source=source,
        source_capsule_bytes_verified_here=authenticated,
        kernel_source_artifact_authenticated_here=False, kernel_flags=deepcopy(numeric.FLAGS),
        candidates=numeric.slots(), zero=np.zeros((5*h, 2), np.float32).tolist(),
        intermediates={}, directions=dict(d0=None, d1=None), direction_reasons=dict(d0=None, d1=None),
        numeric_environment=dict(numpy_version=np.__version__, float64_eps=numeric.EPS,
            float64_dtype=np.dtype(np.float64).str, float32_dtype=np.dtype(np.float32).str),
        resources=None, completed=None, ledger_events=[], error=None)
    on_start(out)
    before, returned, start = deepcopy(ledger.attempted), deepcopy(ledger.completed), len(ledger.events)
    session.event(dict(kind='goal_bank_attempt', local_task_sha256=digest(task), native_response_sha256=digest(response), source=source))

    def generate():
        v = out['intermediates']
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            a, r = q@j, q@(goal-nominal)
            v.update(A=a.tolist(), r=r.tolist())
            b = a.T@r
            require(np.isfinite(b).all(), 'finite projected raw-zero gradient'); v['b'] = b.tolist()
            left, singular, right = ledger.call('array_svd', lambda: np.linalg.svd(a, full_matrices=False))
            n = 10*h
            require(left.shape == (192, n) and singular.shape == (n,) and right.shape == (n, n)
                and all(x.dtype == np.float64 and np.isfinite(x).all() for x in (left, singular, right)), 'full economy float64 SVD')
            v.update(U=left.tolist(), singular_values=singular.tolist(), Vt=right.tolist())
            matrix, rhs = np.diag(singular*singular+numeric.LAMBDA), singular*(left.T@r)
            v.update(solve_matrix=matrix.tolist(), solve_rhs=rhs.tolist())
            w = ledger.call('solve', lambda: np.linalg.solve(matrix, rhs))
            require(w.shape == (n,) and w.dtype == np.float64 and np.isfinite(w).all(), 'finite all-basis solve')
            p = right.T@w; v.update(w=w.tolist(), p=p.tolist())
            directions, reasons, info = numeric.direction_data(p, b)
            out['directions'], out['direction_reasons'] = directions, reasons; v.update(info)
            for row in out['candidates']:
                direction = directions[row['direction']]
                if direction is None:
                    row.update(status='blocked_degenerate_direction', reason=reasons[row['direction']]); continue
                param = row['sign']*row['radius']*np.asarray(direction, np.float64)
                cast = np.tanh(param).astype(np.float32)
                require(np.isfinite(param).all() and np.isfinite(cast).all() and (np.abs(cast) <= 1).all(), 'finite literal actuator transport')
                norm = float(np.linalg.norm(param))
                row.update(status='generated_unexecuted', reason=None, params=param.tolist(), actions=cast.reshape(5*h, 2).tolist(),
                    raw_float32_bits=np.ascontiguousarray(cast, dtype='<f4').tobytes().hex(), param_norm=norm,
                    radius_roundoff_delta=norm-row['radius'], b_dot_param=float(b@param), b_dot_cast32_action=float(b@cast.astype(np.float64)))
    try:
        ledger.call('program_compile', generate)
        out['status'] = 'complete' if all(x['actions'] is not None for x in out['candidates']) else 'complete_with_blocked_directions'
    except Exception as exc:
        out.update(status='partial_failure', error=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        out['resources'] = {k: ledger.attempted[k]-before[k] for k in core.AXES}
        out['completed'] = {k: ledger.completed[k]-returned[k] for k in core.AXES}
        out['ledger_events'] = deepcopy(ledger.events[start:])
        session.event(dict(kind='goal_bank_return', bank=deepcopy(out)))
    require(out['resources'] == out['completed'] == bank_cost(), 'one true SVD/solve/compile; no duplicate kernel ledger')
    return out


def evaluate_R(task, mapping, session, *, on_start):
    ledger, h = session.ledger, mapping['native_response']['horizon']
    before = deepcopy(ledger.attempted)
    require(all(before[k]+group_cost(h)[k]+prior.verification_cost(h)[k] <= ledger.caps[k]
                for k in core.AXES), 'atomic maximum zero+8+bank+whole/segmented reserve')
    g = dict(bank=None, control=dict(index='zero', status='not_requested'),
        rows=[dict(index=i, status='not_requested', eligible=None, actions=None, reason=None) for i in range(8)],
        eligible_indices=[], status='running', resources=None)
    on_start(g)
    bank = make_bank(task, mapping, session, on_start=lambda b: g.update(bank=b))
    for c, row in zip(bank['candidates'], g['rows'], strict=True):
        if c['actions'] is None:
            row.update(status='blocked_degenerate_direction', reason=c['reason'])
            session.event(dict(kind='R_item_blocked', row=deepcopy(row)))
    session.event(dict(kind='R_attempt', parent_context_sha256=digest(task['root_context']), bank_sha256=digest(bank)))
    local = parent.parent_task(task, task['root_context'])
    proposals = [('zero', bank['zero'])]+[(c['index'], c['actions']) for c in bank['candidates'] if c['actions'] is not None]
    for index, raw in proposals:
        row = g['control'] if index == 'zero' else g['rows'][index]
        row.update(status='attempted', actions=deepcopy(raw))
        session.event(dict(kind='R_item_attempt', index=index, parent_context_sha256=digest(task['root_context'])))
        path = ledger.call('program_compile', lambda: ledger.call('replay', lambda: session.replay(local, raw)))
        post = core.checked_path(task, task['root_context'], raw, path)
        row.update(path=path, postcontext=post, immediate=ledger.call('metric', lambda: core.model_cost(task, post)))
        posttask = parent.parent_task(task, post)
        response = ledger.call('signal_compile', lambda: session.signal(posttask, 1))
        nominal, pointer = tree.nominal_path(session.records[-1], posttask, response)
        residual = np.asarray(task['goal'], np.float64)-np.asarray(response['nominal'], np.float64)
        solve = ledger.call('solve', lambda: ledger.call('array_svd', lambda: tree.kernel.original.bounded_ridge(response['jacobian'], residual)))
        suffix = tree.components.suffix_raw(solve['params'])
        suffix_path = ledger.call('replay', lambda: session.replay(posttask, suffix))
        terminal = core.checked_path(task, post, suffix, suffix_path)
        row.update(status='complete', signal=response, nominal_path=nominal, nominal_pointer=pointer, solve=solve,
            reach=solve['own_reach_value'], suffix_actions=suffix, suffix_path=suffix_path,
            bounded_suffix_cost=ledger.call('metric', lambda: core.model_cost(task, terminal)))
        session.event(dict(kind='R_item_return', row=deepcopy(row)))
    zero = g['control']
    for row in g['rows']:
        if row['status'] == 'complete':
            row['eligible'] = row['immediate'] > zero['immediate'] and row['reach'] < zero['reach']
    g['eligible_indices'] = [r['index'] for r in g['rows'] if r['eligible'] is True]
    g['status'] = 'witness_found' if g['eligible_indices'] else 'no_witness'
    g['resources'] = {k: ledger.attempted[k]-before[k] for k in core.AXES}
    require(g['resources'] == group_cost(h, len(proposals)-1), 'actual dispatch census; no padding/refund')
    session.event(dict(kind='R_return', result=deepcopy(g)))
    return g


def delivery(task, pro, h, g, order, verification, source_sha):
    # Reuse the reviewed field builder, explicitly expose a NEW delivery schema
    # and bank authority; never pass this object as a prior producer receipt.
    out = prior.delivery(task, pro, h, g, order, verification, source_sha)
    out.update(schema=NAME+'.delivery', producer_sha256=sha(__file__), protocol_sha256=PROTOCOL_SHA,
        bank_schema=NAME+'.bank', bank_sha256=digest(g['bank']), exploratory_known_roots=True)
    return out


def group(task, pro, h, factory, sink, *, source_sha, permit_ram=False, guard=lambda: None,
          capsule, saved_calls=None, clock=time.monotonic_ns):
    task, pro = input_pair(task, pro); base.signal_cost(h); core.hash_value(source_sha)
    kind = 'RAM_FIXTURE' if permit_ram else 'HOST_CUDA'
    ledger = core.Ledger(caps(h), clock=clock)
    s = tree.Session(task, factory, sink, ledger, kind, guard, capsule, saved_calls); ledger.sink = s.event
    adapter = parent.ParentNativeAdapter(task, lambda t, _: tree.SignalBackend(s, t),
        type('Sink', (), {'event': lambda self, e: s.event(e)})(), permit_ram=permit_ram)
    r = dict(schema=NAME+'.group', backend_kind=kind, horizon=h, task_sha256=digest(task),
        provenance_sha256=digest(pro), source_closure_sha256=source_sha, root_context_role=ROLE,
        actual_cp5_start=True, parent_signal=None, R_groups=[], eligible_order=None,
        delivery=None, model_delivery_verified=False, status='not_started', errors=[],
        verification=[dict(mode=k, segments=[5*h] if k == 'whole' else ([5] if h == 1 else [5, 10]),
                           status='not_requested', call_id=None, path=None) for k in ('whole', 'segmented')], **FLAGS)
    try:
        guard(); ledger.scope = dict(phase='parent_signal', horizon=h, parent_context_sha256=digest(task['root_context']))
        req = core.request(task, parent.root_node(task), operation='compile_finite_horizon_J', horizon=h)
        r['parent_signal'] = ledger.call('signal_compile', lambda: adapter.signal(req, ledger))
        require(ledger.attempted == base.signal_cost(h), 'parent signal exact paid cost')
        ledger.scope = dict(phase='R_group', horizon=h, parent_context_sha256=digest(task['root_context']))
        g = evaluate_R(task, adapter.maps[-1], s, on_start=r['R_groups'].append)
        order = prior.ranked(g); r['eligible_order'] = order
        s.event(dict(kind='delivery_selection', eligible_order=order, selection_key=['immediate', 'candidate_index']))
        if not order:
            r['status'] = 'no_witness'
        else:
            selected = g['rows'][order[0]]
            for v in r['verification']:
                v.update(status='attempted', call_id=len(s.records))
                ledger.scope = dict(phase='delivery_verification', horizon=h, mode=v['mode'], segments=v['segments'])
                s.event(dict(kind='delivery_verification_attempt', verification=deepcopy(v)))
                path = ledger.call('replay', lambda v=v: s.replay(task, selected['actions'], v['mode'] == 'segmented'))
                require(canonical(path) == canonical(selected['path']), 'full edge whole/segmented exact full context/raw bits')
                v.update(status='returned', path=deepcopy(path)); s.event(dict(kind='delivery_verification_return', verification=deepcopy(v)))
            r.update(status='model_delivery_verified', model_delivery_verified=True,
                     delivery=delivery(task, pro, h, g, order, r['verification'], source_sha))
            s.event(dict(kind='model_delivery', delivery=deepcopy(r['delivery'])))
        valid = sum(c['actions'] is not None for c in g['bank']['candidates'])
        require(ledger.attempted == ledger.completed == expected_usage(h, valid, bool(order)), 'all actual charges and returns exact')
        guard()
    except Exception as exc:
        r.update(status='partial_runtime_failure', errors=[dict(type=type(exc).__name__, message=str(exc))],
                 delivery=None, model_delivery_verified=False)
        for g in r['R_groups']:
            if g['status'] == 'running':
                g['status'] = 'partial_runtime_failure'
                for item in [g['control']]+g['rows']:
                    if item['status'] == 'attempted': item['status'] = 'failed'
        for v in r['verification']:
            if v['status'] == 'attempted': v['status'] = 'failed'
    r.update(signal_mappings=deepcopy(adapter.maps), native_calls=deepcopy(s.records), resource_receipt=ledger.receipt())
    core.audit_ledger(r['resource_receipt'])
    return r


def execute(p, factory, store, *, permit_ram=False, check=lambda: None):
    rows = [dict(**x, status='not_requested', result=None) for x in roster()]; failed = False
    store.event(dict(kind='denominator', rows=deepcopy(rows)))
    for row in rows:
        if failed:
            row['status'] = 'blocked_after_runtime_failure'; continue
        slot = row['slot']; begin = getattr(store, 'bytes', 0); start = time.monotonic()
        store.event(dict(kind='group_attempt', identity=roster()[slot]))
        sink = type('Sink', (), {'event': lambda self, e: store.event(dict(kind='group_event', slot=slot, payload=deepcopy(e)))})()
        def guard():
            check(); require(time.monotonic()-start <= p['profile']['group_wall_seconds'], '1800s group wall cap')
        try:
            guard()
            r = group(p['tasks'][row['root']], p['provenance'][row['root']], row['horizon'], factory, sink,
                source_sha=digest(p['sources']), permit_ram=permit_ram, guard=guard,
                capsule=lambda cid, call: store.capsule(slot, cid, call))
            row.update(status=r['status'], result=r, wall_seconds=time.monotonic()-start); failed = bool(r['errors'])
            store.event(dict(kind='group_return', row=deepcopy(row)))
            require(getattr(store, 'bytes', 0)-begin+len(canonical(row)) <= p['profile']['group_output_bytes'], '512MiB group output')
            guard()
        except Exception as exc:
            row.update(status='outer_runtime_failure', error=dict(type=type(exc).__name__, message=str(exc))); failed = True
            store.event(dict(kind='group_outer_failure', row=deepcopy(row)))
    return rows


def own_files():
    return [Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')]


def build():
    require(all(sha(p) == h for p, h in PINS.items()), 'frozen protocol/numerical kernel/native source triples')
    p = prior.build()  # MAIN-only full original actual-cp5 task/source reconstruction; no outcomes.
    require(all(p[k] is False for k in FLAGS), 'real upstream full_C3_demonstrated ABI')
    sources = dict(p['sources']); sources.update({str(p.resolve()): sha(p) for p in [*PINS, *own_files()]})
    return dict(schema=NAME+'.prepare', tasks=p['tasks'], provenance=p['provenance'], cp5_tasks=p['cp5_tasks'],
        policy=p['policy'], expected_model_hash=p['expected_model_hash'], authority=p['authority'],
        sources=sources, roster=roster(), profile=profile(), precision=old.precision_contract(), rules=rules(), **FLAGS)


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
    require(sha(path) == expected, 'actual own prepare SHA'); value = signal.read(path, 16*1024**2)
    require(value == build() and sha(path) == expected, 'full own preparation/source/budget rebuild')
    return value


def run(prepared, expected, output, *, authorization):
    require(authorization == 'MAIN_APPROVED_HOST_CUDA', 'MAIN source audit and HOST CUDA authorization')
    started = time.monotonic(); p = verify(prepared, expected); output = exclusive(output)
    require(shutil.disk_usage(output.parent).free >= p['profile']['output_bytes']+p['profile']['disk_floor_bytes'], '2GiB allowance plus floor')
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
        before = signal.model_hash(model); require(before['original_state_dict'] == p['expected_model_hash'], 'frozen original model')
        classes = old.model_sources(model, p); store.event(dict(kind='model_before', hashes=before, classes=classes))
        forwards = [(m, 'forward' in m.__dict__, m.__dict__.get('forward')) for m in model.modules()]
        with torch.autocast(device_type='cuda', enabled=False), torch.autocast(device_type='cpu', enabled=False), sdpa_kernel(SDPBackend.MATH):
            def guard(): store.check(); old.checked_precision(torch)
            guards.append(old.checked_precision(torch)); store.event(dict(kind='precision_before', value=guards[-1]))
            rows = execute(p, lambda t, s: prior.NativeDelivery(model, device, t, s), store, check=guard)
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
                    and all(not x.requires_grad for x in model.parameters()), 'frozen model and classes beforeafter')
                store.event(dict(kind='model_after', hashes=after, classes=classes_after))
            except BaseException as exc: errors.append(dict(stage='model_final', message=str(exc)))
        try:
            verify(prepared, expected); store.check(); source_ok = True
            store.event(dict(kind='source_after', sha256=digest(p['sources'])))
        except BaseException as exc: errors.append(dict(stage='source_final', message=str(exc)))
        store.close(); settled = all(x['status'] in SETTLED for x in rows)
        report = dict(schema=NAME+'.report', status='four_slots_settled_not_admitted' if settled and not errors else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()), prepare_sha256=expected, rows=rows, errors=errors,
            model_before=before, model_after=after, classes_before=classes, classes_after=classes_after,
            host=host, precision=guards, sources_verified=source_ok, source_manifest_sha256=digest(p['sources']),
            wall_seconds=store.clock()-store.started, CUDA_peak_bytes=store.memory() if store.memory else None,
            journal_tail=store.tail, artifact_sha256={n: sha(output/n) for n in ('binding.json', 'events.jsonl')
                if (output/n).exists()} | store.capsule_hashes, **FLAGS)
        store.put('report.json', report)
    require(report['status'] == 'four_slots_settled_not_admitted', 'partial saved; no retry '+repr(errors))
    return dict(output=str(output), report_sha256=sha(output/'report.json'))


def validate_group(task, pro, h, r, events, *, source_sha, permit_fixture=False, capsule_reader):
    require(r['backend_kind'] == 'HOST_CUDA' or permit_fixture and r['backend_kind'] == 'RAM_FIXTURE', 'no RAM production')
    require(all(e['kind'] == 'io' for e in events), 'own native IO envelopes')
    allowed = old.RESOURCE | old.CALL_ARRAYS | {'direction', 'native_call_attempt', 'native_call_return', 'native_call_failure',
        parent.NAME+'.signal_map', 'goal_bank_attempt', 'goal_bank_return', 'R_item_blocked', 'R_attempt',
        'R_item_attempt', 'R_item_return', 'R_return', 'delivery_selection', 'delivery_verification_attempt',
        'delivery_verification_return', 'model_delivery'}
    require(all(e['payload']['kind'] in allowed for e in events), 'unknown goal R event')
    require(r['resource_receipt']['caps'] == caps(h), 'exact independent caps; only solve+1')
    payloads = [e['payload'] for e in events]; old.resource_census(r['resource_receipt'], payloads)
    calls = r['native_calls']; require([x['call_id'] for x in calls] == list(range(len(calls))), 'native invocation census')
    require(all(e['call'] is None or type(e['call']) is int and 0 <= e['call'] < len(calls) for e in events), 'no orphan call')
    for c in calls:
        require([e['payload'] for e in events if e['call'] == c['call_id']] == c['transcript'], 'native full transcript binding')
        old.audit_native(c['transcript'])
    replay_events = []; sink = type('Sink', (), {'event': lambda self, e: replay_events.append(deepcopy(e))})()
    replay = group(task, pro, h, None, sink, source_sha=source_sha, permit_ram=permit_fixture,
        capsule=capsule_reader, saved_calls=calls, clock=tree.replay_clock(r['resource_receipt'], payloads))
    require(canonical(replay) == canonical(r), 'exact saved full bank/zero/eight/null/selection/delivery/resources')
    require([e for e in replay_events if e['call'] is None] == [e for e in events if e['call'] is None], 'all orchestration events exact order/census')
    complete = r['status'] in SETTLED; has_delivery = r['status'] == 'model_delivery_verified'
    valid = sum(c['actions'] is not None for c in r['R_groups'][0]['bank']['candidates']) if complete else None
    if complete:
        require(len(calls) == 1+3*(1+valid)+2*has_delivery, 'no hidden/unused native calls or blocked slot padding')
        require(r['resource_receipt']['attempted'] == r['resource_receipt']['completed'] == expected_usage(h, valid, has_delivery), 'actual complete charges')
    return dict(horizon=h, status=r['status'], complete=complete, model_delivery_verified=has_delivery,
        valid_candidates=valid, blocked_candidates=8-valid if complete else None,
        eligible_order=r['eligible_order'] if complete else None, no_witness=not has_delivery if complete else None,
        delivery=deepcopy(r['delivery']), native_calls=len(calls), resources=r['resource_receipt'],
        AD_primal_predict_API=h+1+valid if complete else None,
        Jacobian_elements=192*10*(h+1+valid) if complete else None,
        scalar_backward_calls=None, total_FLOPs=None, **FLAGS)


def validate_rows(p, rows, events, *, permit_fixture=False, capsule_reader):
    require(len(rows) == 4 and [(r['slot'], r['root'], r['horizon']) for r in rows] ==
            [(r['slot'], r['root'], r['horizon']) for r in roster()], 'all four fixed slots')
    top = {'denominator', 'group_attempt', 'group_return', 'group_outer_failure', 'source_before', 'source_after',
           'model_before', 'model_after', 'precision_before', 'precision_after'}
    require(all(e['kind'] in top | {'group_event'} for e in events), 'unknown top event')
    require(all(type(e['slot']) is int and 0 <= e['slot'] < 4 for e in events if e['kind'] == 'group_event'), 'orphan slot')
    require([e['rows'] for e in events if e['kind'] == 'denominator'] ==
            [[dict(**x, status='not_requested', result=None) for x in roster()]], 'initial full denominator')
    failed = False; position = -1; out = []
    for row, identity in zip(rows, roster(), strict=True):
        es = [e['payload'] for e in events if e['kind'] == 'group_event' and e['slot'] == row['slot']]
        starts = [i for i, e in enumerate(events) if e['kind'] == 'group_attempt' and e['identity'] == identity]
        ends = [i for i, e in enumerate(events) if e['kind'] == 'group_return' and e['row'] == row]
        if row['result'] is None:
            require(failed and row['status'] == 'blocked_after_runtime_failure' and not starts and not ends and not es, 'runtime-only subsequent stop'); continue
        require(not failed and row['status'] == row['result']['status'] and len(starts) == len(ends) == 1
                and position < starts[0] < ends[0], 'exact dispatch order')
        require(all(starts[0] < i < ends[0] for i, e in enumerate(events) if e['kind'] == 'group_event' and e['slot'] == row['slot']), 'native events in dispatch')
        position = ends[0]
        require(type(row['wall_seconds']) in (int, float) and 0 <= row['wall_seconds'] <= p['profile']['group_wall_seconds'], 'group wall bound')
        summary = validate_group(p['tasks'][row['root']], p['provenance'][row['root']], row['horizon'], row['result'], es,
            source_sha=digest(p['sources']), permit_fixture=permit_fixture,
            capsule_reader=lambda cid, call, slot=row['slot']: capsule_reader(slot, cid, call))
        out.append(dict(slot=row['slot'], root=row['root'], **summary)); failed = bool(row['result']['errors'])
    require(len([e for e in events if e['kind'] == 'group_attempt']) == len(out) == len([e for e in events if e['kind'] == 'group_return']), 'no extra dispatch')
    require(not any(e['kind'] == 'group_outer_failure' for e in events), 'outer failure retained not certified')
    return out


def consume(output, report_sha, prepared, prepare_sha):
    output = Path(output).absolute(); require(output.resolve() == output and not output.is_symlink(), 'exact own output')
    core.hash_value(report_sha); rp = old.regular(output/'report.json', profile()['final_reserve_bytes'])
    require(sha(rp) == report_sha, 'actual report SHA'); r = signal.read(rp, profile()['final_reserve_bytes']); p = verify(prepared, prepare_sha)
    require(r['schema'] == NAME+'.report' and r['prepared'] == str(Path(prepared).absolute())
        and r['prepare_sha256'] == prepare_sha, 'own authority schema')
    require(not r['errors'] and all(r[k] is False for k in FLAGS) and r['sources_verified'] is True
        and r['source_manifest_sha256'] == digest(p['sources']), 'actual source/science closure full_C3_demonstrated')
    require(r['model_before'] == r['model_after'] and r['model_before']['original_state_dict'] == p['expected_model_hash'], 'frozen model')
    require(r['classes_before'] == r['classes_after'] and r['classes_before'], 'installed class closure')
    for c in r['classes_before']: require(p['sources'].get(c['path']) == c['sha256'], 'actual class source pin')
    require(r['precision'] == [old.precision_contract()]*2 and r['host']['device'] == 'cuda:0'
        and r['host']['allocation_sum'] == 4. and r['host']['visible_nvidia_nodes'], 'HOST CUDA and numerical guards')
    names = set(r['artifact_sha256'])
    require({'binding.json', 'events.jsonl'} <= names and all(n in ('binding.json', 'events.jsonl') or
        n.startswith('signal_s') and n.endswith('.json') and Path(n).name == n for n in names), 'explicit artifacts only')
    require({x.name for x in output.iterdir()} == names | {'report.json'}, 'exact all file census')
    for n, h in r['artifact_sha256'].items(): require(sha(old.regular(output/n, profile()['output_bytes'])) == h, 'all saved bytes')
    require(signal.read(output/'binding.json') == dict(prepared=r['prepared'], prepare_sha256=prepare_sha), 'persisted binding')
    sizes = {}; es = base.journal(output, r, sizes)
    for phase in ('before', 'after'):
        require([e for e in es if e['kind'] == 'model_'+phase] == [dict(kind='model_'+phase, hashes=r['model_'+phase], classes=r['classes_'+phase])], 'model journal')
        require([e for e in es if e['kind'] == 'source_'+phase] == [dict(kind='source_'+phase, sha256=digest(p['sources']))], 'source journal')
        require([e['value'] for e in es if e['kind'] == 'precision_'+phase] == [old.precision_contract()], 'precision journal')
    seen = set()
    def capsule_reader(slot, cid, call):
        n = f'signal_s{slot}_{cid:04}.json'; require(n in names, 'actual source signal capsule')
        require(signal.read(old.regular(output/n, 64*1024**2), 64*1024**2) == call and sha(output/n) == digest(call), 'source capsule exact bytes')
        seen.add(n)
        return dict(path=str(output/n), sha256=sha(output/n))
    summaries = validate_rows(p, r['rows'], es, capsule_reader=capsule_reader)
    require(seen == names-{'binding.json', 'events.jsonl'}, 'no dangling capsules')
    require(r['status'] == 'four_slots_settled_not_admitted' and len(summaries) == 4 and all(x['complete'] for x in summaries), 'all four settled including null/no_witness')
    require(0 <= r['wall_seconds'] <= p['profile']['wall_seconds'] and
        sum(x.stat().st_size for x in output.iterdir()) <= p['profile']['output_bytes'], 'wall/output caps')
    require(type(r['CUDA_peak_bytes']) is list and len(r['CUDA_peak_bytes']) == 2 and all(type(v) is int and 0 <= v <= p['profile'][k]
        for v, k in zip(r['CUDA_peak_bytes'], ('cuda_allocated_bytes', 'cuda_reserved_bytes'))), 'CUDA peak caps')
    for row in r['rows']:
        used = sizes[row['slot']]+len(canonical(row))+sum((output/n).stat().st_size for n in names if n.startswith('signal_s'+str(row['slot'])+'_'))
        require(used <= p['profile']['group_output_bytes'], 'full group output bound')
    require(sha(rp) == report_sha and all(sha(f) == h for f, h in p['sources'].items()), 'unchanged final sources and report')
    return dict(schema=NAME+'.consume', status='actual_cp5_goal_R_model_deliveries_verified_NOT_physical',
        prepared=r['prepared'], prepare_sha256=prepare_sha, report_sha256=report_sha, cp5_tasks=p['cp5_tasks'],
        protocol_sha256=PROTOCOL_SHA, denominator=4, candidate_denominator=32, controls=4,
        arms=summaries, root_context_role=ROLE, **FLAGS)


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
