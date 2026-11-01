"""K2 host engine seam; not a complete production adapter or authority."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import time
from experiments.spectral_reachability import agent_p3_c3_k2_common_cap_20261029_v1 as k2

tree, parent, core, live = k2.tree, k2.parent, k2.core, k2.live
old, canonical, require, digest = k2.old, k2.canonical, k2.require, k2.digest
decision, signal_cost, caps = k2.decision, k2.signal_cost, k2.caps
source_closure, MODES, FLAGS, NAME = k2.source_closure, k2.MODES, k2.FLAGS, k2.NAME
K2_SHA = '3b32f147af8fa3baed36634461cd83624cba0900ff7b4fbb37c4cb89ae8f5aec'
AUTH = 'MAIN_APPROVED_K2_HOST_CUDA'


def check_k2_source():
    require(core.sha(Path(k2.__file__).resolve()) == K2_SHA, 'frozen K2 engine source')
    source_closure()


def run_tree(task, factory, sink, *, mode, K, permit_ram=False, kind='HOST_CUDA',
             guard=lambda: None, capsule=None, saved_calls=None, clock=time.monotonic_ns, authorization=None):
    require(K == 2 and ((permit_ram and kind == 'RAM_FIXTURE') or (kind == 'HOST_CUDA' and authorization == AUTH)), 'K2 explicit host authorization or RAM fixture')
    check_k2_source()
    task = core.validate_task(task); tree.components.post_task(task, [], [])
    require(mode in MODES and (kind == 'HOST_CUDA' or permit_ram and kind == 'RAM_FIXTURE'), 'explicit native or RAM')
    require(callable(capsule), 'immutable source capsule callback')
    ledger = core.Ledger(caps(), clock=clock)
    session = tree.Session(task, factory, sink, ledger, kind, guard, capsule, saved_calls)
    ledger.sink = session.event
    adapter = parent.ParentNativeAdapter(task, lambda t, s: tree.SignalBackend(session, t),
        type('Sink', (), {'event': lambda self, e: session.event(e)})(), permit_ram=permit_ram)
    jobs = [dict(slot=i, status='not_requested', request=None, payload=None, node=None, error=None) for i in range(144)]
    r = dict(schema=NAME+'.tree', mode=mode, K=K, backend_kind=kind, root_task_sha256=digest(task),
        jobs=jobs, signals=[], queues=[], R_groups=[],
        replays=[dict(mode=x, status='not_requested', path=None) for x in ('whole', 'segmented')],
        status='running', errors=[], model_plan=None, diagnostic_best=None,
        fixed_denominator=dict(arms=1, requests=144), **FLAGS)
    policy = core.policy(task); frontier = [parent.root_node(task)]; best = None; used = 0; stopped = False
    def rank(n): return n['cost'], n['depth'], n['indices']
    def emit(e): session.event(e)
    try:
        for _ in range(policy.max_depth):
            children = []
            for node in frontier:
                guard(); tree.validate_node(task, node)
                minimum = signal_cost(); remaining = {a: ledger.caps[a]-ledger.attempted[a] for a in core.AXES}
                if any(minimum[a] > remaining[a] for a in core.AXES):
                    emit(dict(kind='signal_refusal', parent=deepcopy(node), minimum=minimum, remaining=remaining))
                    r['status'] = 'resource_cap_before_signal'; stopped = True; break
                ledger.scope = dict(mode=mode, K=K, phase='signal', parent_context_sha256=digest(node['context']))
                signals, receipt = core.local_signal(task, node, adapter, ledger, policy); r['signals'].append(receipt)
                ledger.scope = dict(mode=mode, K=K, phase='schedule', parent_context_sha256=digest(node['context']))
                d = ledger.call('schedule', lambda: decision(task, node, signals, mode))
                reqs = [core.request(task, node, operation='compile_program_extension', **d, candidate_index=i,
                    prefix_edges=deepcopy(node['edges']), prefix_actions=deepcopy(node['actions']),
                    signal_sha256=digest(receipt), local_response=deepcopy(receipt['responses']))
                    for i in range(d['candidates'])] if d['continue_expanding'] else []
                q = dict(parent=deepcopy(node), decision=d, requests=reqs, status='stopped_by_original_child_policy',
                         minimum=None, verification_reserve=None)
                r['queues'].append(q)
                if not reqs: emit(dict(kind='queue', queue=deepcopy(q))); continue
                n, h, kind_ = d['candidates'], d['horizon'], d['extension_type']
                require(n in (4, 8) and h in (1, 3), 'original width and horizon')
                cost = tree.group_cost(h) if kind_ == 'reconfiguration' else parent.generation_minimum(h, n)
                reserve = tree.reserve(node, h); q.update(minimum=cost, verification_reserve=reserve)
                if used+n > 144: q['status'] = 'expansion_cap_before_node_queue'
                elif kind_ not in ('progress', 'reconfiguration') or d['program_type'] not in ('progress', 'reconfiguration', 'composition'):
                    q['status'] = 'refused_unsupported_original_type'
                elif any(ledger.attempted[a]+cost[a]+reserve[a] > ledger.caps[a] for a in core.AXES):
                    q['status'] = 'resource_cap_before_node_queue'
                else: q['status'] = 'queued'
                emit(dict(kind='queue', queue=deepcopy(q)))
                if q['status'] != 'queued': r['status'] = q['status']; stopped = True; break
                assigned = jobs[used:used+n]
                for row, req in zip(assigned, reqs, strict=True): row.update(status='queued', request=deepcopy(req))
                used += n; ledger.scope = dict(mode=mode, K=K, phase='program', parent_context_sha256=digest(node['context']))
                mapping = next(m for m in reversed(adapter.maps) if m['outer_request']['parent_context_sha256'] == digest(node['context'])
                               and m['native_response']['horizon'] == h)
                local = mapping['parent_task']
                if kind_ == 'reconfiguration':
                    g = tree.evaluate_R(task, node, d, mapping, session, reserve, emit, on_start=r['R_groups'].append)
                    proposals = g['rows']
                else:
                    bank = None; proposals = []
                    for req in reqs:
                        def generate():
                            nonlocal bank
                            if bank is None: bank = live.generate(local, mapping['native_response'], n, ledger)
                            raw = bank['actions'][req['candidate_index']]
                            return raw, session.replay(local, raw)
                        raw, path = ledger.call('program_compile', generate)
                        proposals.append(dict(index=req['candidate_index'], actions=raw, path=path, eligible=True,
                            immediate=ledger.call('metric', lambda path=path: core.model_cost(task, path[-1]))))
                    emit(dict(kind='progress_bank', bank=bank, parent_context_sha256=digest(node['context'])))
                for row, req, v in zip(assigned, reqs, proposals, strict=True):
                    c = tree.child(task, node, req, v['actions'], v['path'], kind_); c['cost'] = v['immediate']
                    row.update(status='returned_eligible' if v['eligible'] else 'returned_ineligible', payload=deepcopy(v), node=deepcopy(c))
                    if v['eligible']:
                        children.append(c)
                        if best is None or rank(c) < rank(best): best = deepcopy(c)
            if stopped or not children: break
            children.sort(key=rank)
            frontier = children[:policy.base_beam*(policy.expansion_factor if mode == 'adaptive' else 1)]
            for c in frontier: c.pop('cost')
        if not stopped:
            if best is None: r['status'] = 'no_witness'
            else:
                for slot in r['replays']:
                    slot['status'] = 'attempted'; ledger.scope = dict(mode=mode, K=K, phase='verification', replay_mode=slot['mode'])
                    path = ledger.call('replay', lambda slot=slot: session.replay(task, best['actions'], slot['mode'] == 'segmented'))
                    require(path == best['path'], 'independent whole/segmented full chain exact'); slot.update(status='returned', path=path)
                r.update(status='tree_complete_not_admitted', model_plan=deepcopy(best))
    except Exception as exc:
        r.update(status='failed_or_budget_exhausted'); r['errors'].append(dict(type=type(exc).__name__, message=str(exc)))
        for g in r['R_groups']:
            if g['status'] == 'running':
                g['status'] = 'partial_runtime_failure'
                for item in [g['control']]+g['rows']:
                    if item['status'] == 'attempted': item['status'] = 'failed'
        for row in jobs:
            if row['status'] == 'queued': row.update(status='blocked_after_failure', error=str(exc))
        for slot in r['replays']:
            if slot['status'] == 'attempted': slot['status'] = 'failed'
    r.update(diagnostic_best=best, resource_receipt=ledger.receipt(), signal_mappings=deepcopy(adapter.maps),
        native_calls=deepcopy(session.records), realized_widths=[len(q['requests']) for q in r['queues'] if q['status'] == 'queued'])
    core.audit_ledger(r['resource_receipt'])
    return r

def validate_tree(task, r, events, *, permit_fixture=False, capsule_reader):
    require(r['mode'] in MODES and r['K'] == 2 and r['schema'] == NAME+'.tree', 'own policy/budget schema')
    require(r['backend_kind'] == 'HOST_CUDA' or permit_fixture and r['backend_kind'] == 'RAM_FIXTURE', 'no RAM production receipt')
    require(all(e['kind'] == 'io' for e in events), 'own IO envelope')
    payloads = [e['payload'] for e in events]
    allowed = old.RESOURCE | old.CALL_ARRAYS | {'direction', 'native_call_attempt', 'native_call_return', 'native_call_failure',
        parent.NAME+'.signal_map', 'queue', 'bank_event', 'R_attempt', 'R_item_attempt', 'R_item_return', 'R_return', 'progress_bank', 'signal_refusal'}
    require(all(e['kind'] in allowed for e in payloads), 'unknown common-cap event')
    require(r['resource_receipt']['caps'] == caps(), 'identical approved per-K vector')
    old.resource_census(r['resource_receipt'], payloads)
    records = r['native_calls']; require([x['call_id'] for x in records] == list(range(len(records))), 'complete native invocation IDs')
    require(all(e['call'] is None or type(e['call']) is int and 0 <= e['call'] < len(records) for e in events), 'no orphan native event')
    for record in records:
        require([e['payload'] for e in events if e['call'] == record['call_id']] == record['transcript'], 'entire native transcript')
        old.audit_native(record['transcript'])
    es = []; sink = type('Sink', (), {'event': lambda self, e: es.append(deepcopy(e))})()
    replay = run_tree(task, None, sink, mode=r['mode'], K=r['K'], permit_ram=permit_fixture, kind=r['backend_kind'],
        capsule=capsule_reader, saved_calls=records, clock=tree.replay_clock(r['resource_receipt'], payloads), authorization=AUTH if r['backend_kind']=='HOST_CUDA' else None)
    require(canonical(replay) == canonical(r), 'complete saved policy/signal/bank/R/beam/cap/replay reconstruction')
    require([e for e in es if e['call'] is None] == [e for e in events if e['call'] is None], 'whole noncall event sequence')
    require(r['model_plan'] is None or r['status'] == 'tree_complete_not_admitted', 'no partial best-plan release')
    calls = [x['transcript'] for x in records if x['transcript'][0]['method'] == 'signal' and x['transcript'][-1]['kind'] == 'native_call_return']
    return dict(mode=r['mode'], K=r['K'], status=r['status'], plan_available=r['model_plan'] is not None,
        R_groups=len(r['R_groups']), eligible_counts=[len(g['eligible_indices']) for g in r['R_groups']],
        native_calls=len(records), resources=r['resource_receipt'],
        AD_primal_predict_API=sum(c[0]['request']['horizon'] for c in calls),
        Jacobian_elements=sum(1920*c[0]['request']['horizon'] for c in calls),
        scalar_backward_calls=None, total_FLOPs=None, **FLAGS)


