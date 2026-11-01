"""Common-cap twelve-slot native policy experiment; only MAIN may run HOST CUDA."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time

from experiments.spectral_reachability import agent_p3_r_native_orchestrator_20261029_v1 as tree

old, parent, core, live, signal = tree.old, tree.parent, tree.core, tree.live, tree.signal
require, canonical, digest, sha = core.require, core.canonical, core.digest, core.sha
ROOT = core.ROOT
NAME = 'agent_p3_c3_common_cap_cuda_20261029_v1'
PROTOCOL = ROOT/'innerdocs/agent_p3_c3_common_cap_scheduler_protocol_20261029_v1.md'
PROTOCOL_SHA = '0ea4114c0bc88f780f2a788644ddc24851e7ba32e129029350cc5c98bad2255d'
PINS = {Path(tree.__file__).resolve(): 'da00e09c89ed8b5be5399952c2ec328a98e036db08f764b61342d69a18f5e6a6',
    Path(parent.__file__).resolve(): '150a1c4189a68f9b7c71479e5aa6980cea490fdeed0d5524b7bfad0cc78c2558',
    Path(core.__file__).resolve(): '72f076106fb8f6c2aa73386c06fecb912f033e81a5108aa1f36885bd0342c2bc',
    Path(tree.spectral.__file__).resolve(): 'e254f08db87b1e8b6985b6afd6ba3f3ed8f98e8f1af0d87f9edcd14021f2786d',
    PROTOCOL: PROTOCOL_SHA}
MODES = ('fixedH1', 'staticH3_progress', 'adaptive')
FLAGS = dict(tree.FLAGS, same_actual_resources=False, same_FLOPs=False, physical_release=False)
SETTLED = {'tree_complete_not_admitted', 'no_witness', 'resource_cap_before_signal',
           'resource_cap_before_node_queue', 'expansion_cap_before_node_queue', 'refused_unsupported_original_type'}


def roster():
    return [dict(slot=i, root=r, mode=m, K=k) for i, (r, k, m) in
            enumerate((r, k, m) for r in (0, 1) for k in (1, 17) for m in MODES)]


def signal_cost():
    v = live.predict_axes(8)
    v.update(jacobian_API=2, signal_compile_calls=2, VJP_directions=384, grad_API=384,
             raw_backward_callbacks=384, array_SVD_calls=2, schedule_decisions=1)
    return v


def caps(k):
    require(type(k) is int and k in (1, 17), 'fixed K1/K17')
    v = tree.sum_axes(signal_cost(), tree.group_cost(3))
    v = {a: k*n for a, n in v.items()}
    v = tree.sum_axes(v, dict(live.predict_axes(18), replay_calls=2))
    v['schedule_decisions'] = 17
    return v


def proof():
    """Full-shape normative bounds, never a measured NN/output/time certificate."""
    signal_ = signal_cost(); worst = tree.group_cost(3)
    for h in (1, 3):
        for w in (4, 8):
            require(all(parent.generation_minimum(h, w)[a] <= worst[a] for a in core.AXES), 'progress dominated by R3')
        require(all(tree.group_cost(h)[a] <= worst[a] for a in core.AXES), 'R1 dominated by R3')
    require((caps(1)['predict_API'], caps(1)['jacobian_API'], caps(17)['predict_API'], caps(17)['jacobian_API'])
            == (80, 11, 1072, 187), 'approved full-shape counts')
    return dict(max_parents=17, max_external_requests=136, identity_denominator=144,
        full_context_shapes=dict(latent=[3, 192], history=[2, 10]),
        parent_J_shapes=[[192, 10], [192, 30]], post_J_shape=[192, 10],
        max_edge_raw_shape=[15, 2], max_chain_raw_shape=[45, 2], suffix_raw_shape=[5, 2],
        signal=signal_, R3=worst, common_caps={str(k): caps(k) for k in (1, 17)},
        parent_AD_primal_predict=4, parent_J_elements=7680, post_AD_primal_predict=1, post_J_elements=1920,
        numerical_cost_bound_proved=True, serialized_output_worstcase_fit_proved=byte_census()['fits_live_allowance'],
        serialized_output_census=byte_census(),
        wall_worstcase_fit_proved=False, CUDA_memory_worstcase_fit_proved=False,
        output_semantics='hard live cap, sharded full slot results; overflow is retained failure, not padding/refund',
        scalar_backward_calls=None, total_FLOPs=None)


def byte_census():
    """Conservative format-envelope bound, NOT a claim of actual output size.

    Shapes alone do not bound repeated literal task/source dictionaries. Include
    their actual parser/serializer envelopes, rather than pricing just arrays.
    This deliberately loose bound FAILS the approved allowance. No NN needed.
    """
    limits = profile(); slots = []
    for identity in roster():
        c = caps(identity['K']); j = c['jacobian_API']; v = c['VJP_directions']; n = c['predict_API']
        # Full3: <=8192 numeric elements across packet/boundary and six model
        # input/output buffers per predict. VJP: basis192, terminal192, raw30,
        # returned30, raw-backward30. J return has <=192*30+192 elements.
        elements = n*8192 + v*(192+192+30+30+30) + j*(192*30+192)
        # Canonical finite Python binary64 repr <=24 ASCII bytes, plus comma.
        # Native blob is base64 of <=8-byte elements (both f4/f8 accounted).
        numeric = elements*25 + 4*((elements*8+2)//3)
        # Each native predict/VJP contributes fewer than 32 records; include
        # every ledger attempt/return, call delimiters, 144 queues/jobs, bank
        # and parent maps. 128 per native signal covers non-direction records.
        event_count = 32*(n+v)+128*j+256*144+4096
        # Exact format envelopes are enforced on writes as well as reads.
        # journal includes all native_calls again; capsule repeats at most one
        # chosen parent signal per R parent, <=17 (1+8+8). Full slot shard
        # includes native_calls, jobs, parent maps, bank/source metadata.
        parts = dict(events=event_count*limits['event_read_cap_bytes'],
                     capsules=17*64*1024**2, shard=limits['result_read_cap_bytes'])
        slots.append(dict(**identity, numeric_element_upper=elements,
            numeric_JSON_and_blob_bytes_upper=numeric, event_records_upper=event_count,
            native_calls_upper=j+c['replay_calls']+144,
            capsule_count_upper=17, parts_bytes_upper=parts,
            bytes_upper=sum(parts.values())))
    # binding and all final metadata together must fit reserved512MiB. Native
    # numeric estimate is explanatory only: it is ALREADY in the envelopes,
    # not an extra allocation and not silently substituted for metadata.
    live_upper = sum(s['bytes_upper'] for s in slots)
    allowance = limits['output_bytes']-limits['final_reserve_bytes']
    return dict(slots=slots, fixed_denominator=12, JSON_float_token_bytes_upper=24,
        per_float_with_comma_bytes_upper=25, native_blob_element_bytes_upper=8,
        full_shapes=dict(latent=[3,192], action_history=[2,10], J=[[192,10],[192,30]],
                         edge=[15,2], chain=[45,2], suffix=[5,2]),
        duplication='journal + native_calls in shard + up to17 full signal capsules; shard also retains maps/jobs/banks',
        metadata='included in enforced whole-record/capsule/shard envelopes; not assumed zero or average-sized',
        live_bytes_upper=live_upper, live_allowance_bytes=allowance,
        final_metadata_bytes_upper=limits['final_reserve_bytes'],
        total_bytes_upper=live_upper+limits['final_reserve_bytes'],
        fits_live_allowance=live_upper <= allowance,
        bound_kind='conservative enforced format envelopes, NOT a tight shape-only serialization proof',
        production_blocker='upper bound exceeds approved allowance; no HOST model loading authorized by this artifact')


def require_output_fit():
    require(byte_census()['fits_live_allowance'], 'preflight serialized upper bound exceeds 4GiB minus512MiB; production blocked')


def profile():
    return dict(output_bytes=4*1024**3, final_reserve_bytes=512*1024**2,
        wall_seconds=7200, slot_wall_seconds=900, disk_floor_bytes=200*1024**2,
        cuda_allocated_bytes=3*1024**3, cuda_reserved_bytes=4*1024**3,
        max_requests=144, encoder_calls=0, physics_steps=0, CPU_neural=False,
        resource_caps={str(k): caps(k) for k in (1, 17)}, no_padding=True, no_retry=True,
        result_read_cap_bytes=4*1024**3-512*1024**2, event_read_cap_bytes=128*1024**2)


def rules():
    return dict(protocol_sha256=PROTOCOL_SHA, roster=roster(), same_cap_vector=True,
        zero_and_all8_R_paid=True, all_eligible_to_original_modelQ_beam=True,
        unchosen_horizon_candidate_panels=False, unsupported_fallback=False,
        context_role='original known root; NOT actual-cp5; model child is not real feedback',
        progress_bank='original fractions', R_bank='frozen spectral8; no secant/goalbank',
        partial_model_plan_release=False, scientific_threshold_changes=False,
        stopping='cap/unsupported stops tree; no_witness stops branch; runtime failure blocks later slots', **FLAGS)


def decision(task, node, signals, mode):
    require(mode in MODES, 'three fixed policy identities')
    d = core.choose(task, node, signals, 'adaptive' if mode == 'adaptive' else 'fixed')
    if mode == 'staticH3_progress':
        d = dict(d, horizon=3, decision_source=NAME+'.staticH3_progress: declared static counterfactual')
    return d


def run_tree(task, factory, sink, *, mode, K, permit_ram=False, kind='HOST_CUDA',
             guard=lambda: None, capsule=None, saved_calls=None, clock=time.monotonic_ns):
    task = core.validate_task(task); tree.components.post_task(task, [], [])
    require(mode in MODES and (kind == 'HOST_CUDA' or permit_ram and kind == 'RAM_FIXTURE'), 'explicit native or RAM')
    require(callable(capsule), 'immutable source capsule callback')
    ledger = core.Ledger(caps(K), clock=clock)
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


class Store(tree.Store):
    """Full per-slot JSON shards; no tree duplicated in top-level return event/report."""
    def __init__(self, output, limits): super().__init__(output, limits); self.slot_hashes = {}
    def event(self, value):
        record = dict(index=self.index, previous=self.tail, payload=value)
        require(len(canonical(dict(record, sha256=digest(record))))+1 <= self.limits['event_read_cap_bytes'], 'bounded complete event before write')
        return super().event(value)
    def capsule(self, slot, cid, call):
        require(len(canonical(call)) <= 64*1024**2, 'bounded source capsule before write')
        return super().capsule(slot, cid, call)
    def result(self, slot, value):
        name = f'slot_{slot:02}.json'; raw = canonical(value)
        require(len(raw) <= self.limits['result_read_cap_bytes'], 'bounded full slot result')
        self.check(len(raw)); path = self.output/name
        with path.open('xb') as f:
            require(f.write(raw) == len(raw), 'full slot shard'); f.flush(); os.fsync(f.fileno())
        self.bytes += len(raw); signal.fsync_dir(self.output); self.slot_hashes[name] = sha(path)
        return dict(name=name, sha256=self.slot_hashes[name])


def persist_completion(store, report):
    """Witness report/file+directory fsync return; completion's own I/O excluded.

    No finite file can record a clock sample taken after its OWN final fsync.
    The explicit boundary is therefore report durable, witnessed independently.
    A post-completion runtime check additionally refuses successful return if
    completion persistence itself crosses the cap (not a saved timing proof).
    """
    report_bytes = len(canonical(report))
    binding_bytes = (store.output/'binding.json').stat().st_size if (store.output/'binding.json').exists() else 0
    require(report_bytes+binding_bytes <= store.limits['final_reserve_bytes'], 'final metadata reserve')
    store.put('report.json', report)
    elapsed = store.clock()-store.started
    artifacts = {x.name: dict(sha256=sha(x), bytes=x.stat().st_size) for x in store.output.iterdir()}
    witness = dict(schema=NAME+'.completion', boundary='report_file_and_directory_fsync_returned',
        excludes='completion.json own serialization/write/fsync and subsequent process shutdown',
        report_sha256=sha(store.output/'report.json'), journal_tail=store.tail,
        wall_seconds_through_report_fsync=elapsed, artifacts=artifacts,
        within_wall=0 <= elapsed <= store.limits['wall_seconds'])
    raw = canonical(witness)
    require(report_bytes+binding_bytes+len(raw) <= store.limits['final_reserve_bytes'], 'combined final metadata reserve')
    store.check(len(raw), final=True)
    with (store.output/'completion.json').open('xb') as f:
        require(f.write(raw) == len(raw), 'complete completion witness'); f.flush(); os.fsync(f.fileno())
    store.bytes += len(raw); signal.fsync_dir(store.output)
    require(witness['within_wall'] and store.clock()-store.started <= store.limits['wall_seconds'], 'completion persistence wall cap')
    return witness


def validate_completion(output, report, report_sha, limits):
    path = old.regular(output/'completion.json', limits['final_reserve_bytes'])
    w = signal.read(path, limits['final_reserve_bytes'])
    core.keys(w, ('schema', 'boundary', 'excludes', 'report_sha256', 'journal_tail',
                  'wall_seconds_through_report_fsync', 'artifacts', 'within_wall'))
    require(w['schema'] == NAME+'.completion' and w['boundary'] == 'report_file_and_directory_fsync_returned'
        and w['excludes'] == 'completion.json own serialization/write/fsync and subsequent process shutdown', 'explicit durable completion boundary')
    require(w['report_sha256'] == report_sha and w['journal_tail'] == report['journal_tail'], 'completion report/journal binding')
    require(set(w['artifacts']) == set(report['artifact_sha256']) | {'report.json'}, 'complete completion artifact census')
    for n, item in w['artifacts'].items():
        core.keys(item, ('sha256', 'bytes')); f = old.regular(output/n, limits['output_bytes'])
        require(item == dict(sha256=sha(f), bytes=f.stat().st_size), 'completion exact artifact bytes')
    elapsed = w['wall_seconds_through_report_fsync']
    require(type(elapsed) in (int, float) and type(report['wall_seconds']) in (int, float)
        and 0 <= report['wall_seconds'] <= elapsed <= limits['wall_seconds'] and w['within_wall'] is True, 'post-report-fsync wall bound')
    return w


def execute(p, factory, store, *, permit_ram=False, check=lambda: None, on_start=lambda rows: None):
    rows = [dict(**x, status='not_requested', result=None) for x in roster()]; failed = False
    on_start(rows)  # preserve mutated twelve-slot prefix even if the journal fails.
    store.event(dict(kind='denominator', rows=deepcopy(rows), request_slots_per_arm=144))
    for row, identity in zip(rows, roster(), strict=True):
        if failed: row['status'] = 'blocked_after_runtime_failure'; continue
        slot = row['slot']; start = time.monotonic(); store.event(dict(kind='arm_attempt', identity=identity))
        sink = type('Sink', (), {'event': lambda self, e: store.event(dict(kind='arm_event', slot=slot, payload=deepcopy(e)))})()
        def guard():
            check(); require(time.monotonic()-start <= p['profile']['slot_wall_seconds'], 'slot900 wall cap')
        try:
            guard()
            r = run_tree(p['tasks'][row['root']], factory, sink, mode=row['mode'], K=row['K'], permit_ram=permit_ram,
                kind='RAM_FIXTURE' if permit_ram else 'HOST_CUDA', guard=guard,
                capsule=lambda cid, call: store.capsule(slot, cid, call))
            ref = store.result(slot, r); row.update(status=r['status'], result=ref, wall_seconds=time.monotonic()-start)
            failed = bool(r['errors']); guard()
        except Exception as exc:
            row.update(status='outer_runtime_failure', error=dict(type=type(exc).__name__, message=str(exc))); failed = True
        store.event(dict(kind='arm_return', row=deepcopy(row)))
    return rows


def validate_tree(task, r, events, *, permit_fixture=False, capsule_reader):
    require(r['mode'] in MODES and r['K'] in (1, 17) and r['schema'] == NAME+'.tree', 'own policy/budget schema')
    require(r['backend_kind'] == 'HOST_CUDA' or permit_fixture and r['backend_kind'] == 'RAM_FIXTURE', 'no RAM production receipt')
    require(all(e['kind'] == 'io' for e in events), 'own IO envelope')
    payloads = [e['payload'] for e in events]
    allowed = old.RESOURCE | old.CALL_ARRAYS | {'direction', 'native_call_attempt', 'native_call_return', 'native_call_failure',
        parent.NAME+'.signal_map', 'queue', 'bank_event', 'R_attempt', 'R_item_attempt', 'R_item_return', 'R_return', 'progress_bank', 'signal_refusal'}
    require(all(e['kind'] in allowed for e in payloads), 'unknown common-cap event')
    require(r['resource_receipt']['caps'] == caps(r['K']), 'identical approved per-K vector')
    old.resource_census(r['resource_receipt'], payloads)
    records = r['native_calls']; require([x['call_id'] for x in records] == list(range(len(records))), 'complete native invocation IDs')
    require(all(e['call'] is None or type(e['call']) is int and 0 <= e['call'] < len(records) for e in events), 'no orphan native event')
    for record in records:
        require([e['payload'] for e in events if e['call'] == record['call_id']] == record['transcript'], 'entire native transcript')
        old.audit_native(record['transcript'])
    es = []; sink = type('Sink', (), {'event': lambda self, e: es.append(deepcopy(e))})()
    replay = run_tree(task, None, sink, mode=r['mode'], K=r['K'], permit_ram=permit_fixture, kind=r['backend_kind'],
        capsule=capsule_reader, saved_calls=records, clock=tree.replay_clock(r['resource_receipt'], payloads))
    require(canonical(replay) == canonical(r), 'complete saved policy/signal/bank/R/beam/cap/replay reconstruction')
    require([e for e in es if e['call'] is None] == [e for e in events if e['call'] is None], 'whole noncall event sequence')
    require(r['model_plan'] is None or r['status'] == 'tree_complete_not_admitted', 'no partial best-plan release')
    calls = [x['transcript'] for x in records if x['transcript'][0]['method'] == 'signal' and x['transcript'][-1]['kind'] == 'native_call_return']
    return dict(mode=r['mode'], K=r['K'], status=r['status'], plan_available=r['model_plan'] is not None,
        R_groups=len(r['R_groups']), eligible_counts=[len(g['eligible_indices']) for g in r['R_groups']],
        native_calls=len(records), resources=r['resource_receipt'],
        AD_primal_predict_API=sum(c[0]['request']['horizon'] for c in calls),
        Jacobian_elements=sum(1920*c[0]['request']['horizon'] for c in calls), scalar_backward_calls=None, total_FLOPs=None, **FLAGS)


def validate_rows(p, rows, events, result_reader, capsule_reader, *, permit_fixture=False):
    require(len(rows) == 12 and [{k: r[k] for k in ('slot', 'root', 'mode', 'K')} for r in rows] == roster(), 'complete ordered12 denominator')
    top = {'denominator', 'arm_attempt', 'arm_return', 'source_before', 'source_after', 'model_before', 'model_after', 'precision_before', 'precision_after'}
    require(all(e['kind'] in top | {'arm_event'} for e in events), 'unknown top event')
    require(all(type(e['slot']) is int and 0 <= e['slot'] < 12 for e in events if e['kind'] == 'arm_event'), 'orphan slot')
    initial = [dict(**x, status='not_requested', result=None) for x in roster()]
    require([e for e in events if e['kind'] == 'denominator'] == [dict(kind='denominator', rows=initial, request_slots_per_arm=144)], 'original full denominator')
    summaries = []; failed = False; position = -1
    for row, identity in zip(rows, roster(), strict=True):
        es = [e['payload'] for e in events if e['kind'] == 'arm_event' and e['slot'] == row['slot']]
        starts = [i for i, e in enumerate(events) if e['kind'] == 'arm_attempt' and e['identity'] == identity]
        ends = [i for i, e in enumerate(events) if e['kind'] == 'arm_return' and e['row'] == row]
        if row['result'] is None:
            require(failed and row['status'] == 'blocked_after_runtime_failure' and not starts and not ends and not es, 'runtime-only later block'); continue
        require(not failed and len(starts) == len(ends) == 1 and position < starts[0] < ends[0], 'ordered exact dispatch')
        require(all(starts[0] < i < ends[0] for i, e in enumerate(events) if e['kind'] == 'arm_event' and e['slot'] == row['slot']), 'slot journal interval')
        position = ends[0]; r = result_reader(row['slot'], row['result'])
        require(r['mode'] == row['mode'] and r['K'] == row['K'] and r['status'] == row['status'], 'slot shard identity/status')
        require(type(row['wall_seconds']) in (int, float) and 0 <= row['wall_seconds'] <= p['profile']['slot_wall_seconds'], 'slot wall bounded')
        summary = validate_tree(p['tasks'][row['root']], r, es, permit_fixture=permit_fixture,
            capsule_reader=lambda cid, call, slot=row['slot']: capsule_reader(slot, cid, call))
        summaries.append(dict(slot=row['slot'], root=row['root'], **summary)); failed = bool(r['errors'])
    require(len([e for e in events if e['kind'] == 'arm_attempt']) == len(summaries) == len([e for e in events if e['kind'] == 'arm_return']), 'no hidden or missing slot dispatch')
    return summaries


def own_files():
    return [Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')]


def build():
    require(all(sha(p) == h for p, h in PINS.items()), 'sealed producer and approved common-cap protocol')
    p = tree.build()  # saved original task/source discovery, MAIN only; not worker RAM.
    sources = dict(p['sources']); sources.update({str(f.resolve()): sha(f) for f in [*PINS, *own_files()]})
    return dict(schema=NAME+'.prepare', tasks=p['tasks'], policy=p['policy'], authority=p['authority'],
        expected_model_hash=p['expected_model_hash'], sources=sources, roster=roster(), profile=profile(),
        rules=rules(), shape_and_cost_proof=proof(), precision=old.precision_contract(), **FLAGS)


def exclusive(path):
    p = Path(path).absolute()
    require(p.parent == ROOT/'results' and p.name.startswith(NAME) and p.resolve() == p and not p.exists() and not p.is_symlink(), 'exclusive new own result')
    return p


def write_new(path, value):
    p = exclusive(path); raw = canonical(value)
    require(len(raw) <= profile()['final_reserve_bytes'] and shutil.disk_usage(p.parent).free >= profile()['disk_floor_bytes']+len(raw), 'bounded metadata/floor')
    with p.open('xb') as f:
        require(f.write(raw) == len(raw), 'complete new metadata'); f.flush(); os.fsync(f.fileno())
    signal.fsync_dir(p.parent); return dict(path=str(p), sha256=sha(p))


def prepare(output): return write_new(output, build())


def verify(prepared, expected):
    core.hash_value(expected); path = old.regular(prepared, 16*1024**2)
    require(sha(path) == expected, 'actual prepare SHA'); p = signal.read(path)
    require(p == build() and sha(path) == expected, 'complete authority/source/policy/cap reconstruction')
    return p


def run(prepared, expected, output, *, authorization):
    require(authorization == 'MAIN_APPROVED_HOST_CUDA', 'MAIN audit and serial HOST CUDA only')
    require_output_fit()  # before authority read, output creation, torch/model.
    start = time.monotonic(); p = verify(prepared, expected); output = exclusive(output)
    require(shutil.disk_usage(output.parent).free >= p['profile']['output_bytes']+p['profile']['disk_floor_bytes'], '4GiB allowance plus floor')
    store = Store(output, p['profile']); store.started = start
    rows = [dict(**x, status='blocked_before_model', result=None) for x in roster()]
    model = None; before = after = classes = classes_after = host = None; guards = []; errors = []; source_ok = False
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
        before = signal.model_hash(model); require(before['original_state_dict'] == p['expected_model_hash'], 'original model hash')
        classes = old.model_sources(model, p); store.event(dict(kind='model_before', hashes=before, classes=classes))
        forwards = [(m, 'forward' in m.__dict__, m.__dict__.get('forward')) for m in model.modules()]
        with torch.autocast(device_type='cuda', enabled=False), torch.autocast(device_type='cpu', enabled=False), sdpa_kernel(SDPBackend.MATH):
            def guard(): store.check(); old.checked_precision(torch)
            guards.append(old.checked_precision(torch)); store.event(dict(kind='precision_before', value=guards[-1]))
            def retain_rows(value):
                nonlocal rows
                rows = value
            rows = execute(p, lambda t, s: old.NativeTree(model, device, t, s), store, check=guard, on_start=retain_rows)
            guards.append(old.checked_precision(torch)); store.event(dict(kind='precision_after', value=guards[-1]))
        torch.cuda.synchronize(device)
        require(all(('forward' in m.__dict__) == had and m.__dict__.get('forward') is v for m, had, v in forwards), 'all instance forward hooks restored')
    except BaseException as exc: errors.append(dict(stage='run', type=type(exc).__name__, message=str(exc)))
    finally:
        if model is not None:
            try:
                after = signal.model_hash(model); classes_after = old.model_sources(model, p)
                require(before == after and classes == classes_after and all(not m.training for m in model.modules())
                    and all(not v.requires_grad for v in model.parameters()), 'model/classes frozen beforeafter')
                store.event(dict(kind='model_after', hashes=after, classes=classes_after))
            except BaseException as exc: errors.append(dict(stage='model_final', message=str(exc)))
        try:
            verify(prepared, expected); store.check(); source_ok = True; store.event(dict(kind='source_after', sha256=digest(p['sources'])))
        except BaseException as exc: errors.append(dict(stage='source_final', message=str(exc)))
        store.close(); settled = all(x['status'] in SETTLED and x['result'] is not None for x in rows)
        report = dict(schema=NAME+'.report', status='twelve_slots_settled_not_admitted' if settled and not errors else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()), prepare_sha256=expected, rows=rows, errors=errors,
            model_before=before, model_after=after, classes_before=classes, classes_after=classes_after, host=host,
            precision=guards, sources_verified=source_ok, source_manifest_sha256=digest(p['sources']),
            wall_seconds=store.clock()-store.started, CUDA_peak_bytes=store.memory() if store.memory else None,
            journal_tail=store.tail, artifact_sha256={n: sha(output/n) for n in ('binding.json', 'events.jsonl') if (output/n).exists()}
                | store.capsule_hashes | store.slot_hashes, **FLAGS)
        persist_completion(store, report)
    require(report['status'] == 'twelve_slots_settled_not_admitted', 'partial saved, no retry '+repr(errors))
    return dict(output=str(output), report_sha256=sha(output/'report.json'))


def read_events(output, report):
    path = old.regular(output/'events.jsonl', profile()['output_bytes']); tail = '0'*64; events = []
    with path.open('rb') as f:
        for i, line in enumerate(f):
            require(line.endswith(b'\n') and len(line) <= profile()['event_read_cap_bytes'], 'bounded complete event record')
            record = json.loads(line, object_pairs_hook=signal.unique_pairs); core.keys(record, ('index', 'previous', 'payload', 'sha256'))
            h = record.pop('sha256')
            require(type(record['index']) is int and record['index'] == i and record['previous'] == tail and digest(record) == h, 'actual journal chain')
            tail = h; events.append(record['payload'])
    require(tail == report['journal_tail'], 'exact final journal tail'); return events


def consume(output, report_sha, prepared, prepare_sha):
    output = Path(output).absolute(); require(output.resolve() == output and not output.is_symlink(), 'exact output directory')
    core.hash_value(report_sha); rp = old.regular(output/'report.json', profile()['final_reserve_bytes'])
    require(sha(rp) == report_sha, 'actual report SHA'); r = signal.read(rp, profile()['final_reserve_bytes']); p = verify(prepared, prepare_sha)
    require(r['schema'] == NAME+'.report' and r['prepared'] == str(Path(prepared).absolute()) and r['prepare_sha256'] == prepare_sha, 'own authority schema')
    require(not r['errors'] and all(r[k] is False for k in FLAGS) and r['sources_verified'] is True
        and r['source_manifest_sha256'] == digest(p['sources']), 'engineering closure, not science admission')
    require(r['model_before'] == r['model_after'] and r['model_before']['original_state_dict'] == p['expected_model_hash'], 'frozen model')
    require(r['classes_before'] == r['classes_after'] and r['classes_before'], 'installed class beforeafter')
    for c in r['classes_before']: require(p['sources'].get(c['path']) == c['sha256'], 'installed class pins')
    require(r['precision'] == [old.precision_contract()]*2 and r['host']['device'] == 'cuda:0'
        and r['host']['allocation_sum'] == 4. and r['host']['visible_nvidia_nodes'], 'actual HOST precision')
    names = set(r['artifact_sha256']); expected_slots = {f'slot_{i:02}.json' for i in range(12)}
    require({'binding.json', 'events.jsonl'} | expected_slots <= names and all(n in {'binding.json', 'events.jsonl'} | expected_slots
        or n.startswith('signal_s') and n.endswith('.json') and Path(n).name == n for n in names), 'only own exact artifact names')
    require({x.name for x in output.iterdir()} == names | {'report.json', 'completion.json'}, 'complete file census')
    for n, h in r['artifact_sha256'].items(): require(sha(old.regular(output/n, p['profile']['output_bytes'])) == h, 'all artifact bytes')
    require(signal.read(output/'binding.json') == dict(prepared=r['prepared'], prepare_sha256=prepare_sha), 'persisted binding')
    completion = validate_completion(output, r, report_sha, p['profile'])
    events = read_events(output, r)
    for phase in ('before', 'after'):
        require([e for e in events if e['kind'] == 'model_'+phase] == [dict(kind='model_'+phase, hashes=r['model_'+phase], classes=r['classes_'+phase])], 'model journal')
        require([e for e in events if e['kind'] == 'source_'+phase] == [dict(kind='source_'+phase, sha256=digest(p['sources']))], 'source journal')
        require([e['value'] for e in events if e['kind'] == 'precision_'+phase] == [old.precision_contract()], 'precision journal')
    seen = set()
    def result_reader(slot, ref):
        n = f'slot_{slot:02}.json'
        require(ref == dict(name=n, sha256=r['artifact_sha256'][n]), 'slot identity and exact shard SHA')
        seen.add(n); return signal.read(old.regular(output/n, p['profile']['result_read_cap_bytes']), p['profile']['result_read_cap_bytes'])
    def capsule_reader(slot, cid, call):
        n = f'signal_s{slot}_{cid:04}.json'; require(n in names, 'explicit source capsule')
        require(signal.read(old.regular(output/n, 64*1024**2), 64*1024**2) == call and sha(output/n) == digest(call), 'full original source capsule bytes')
        seen.add(n); return dict(path=str(output/n), sha256=sha(output/n))
    summaries = validate_rows(p, r['rows'], events, result_reader, capsule_reader)
    require(len(summaries) == 12 and all(a['status'] in SETTLED for a in summaries)
        and r['status'] == 'twelve_slots_settled_not_admitted', 'all12 settled; cap/refusal not omitted')
    require(seen == names-{'binding.json', 'events.jsonl'}, 'no unused or aliased slot/capsule')
    require(0 <= r['wall_seconds'] <= p['profile']['wall_seconds'] and sum(x.stat().st_size for x in output.iterdir()) <= p['profile']['output_bytes'], 'whole wall/output bounds')
    require(type(r['CUDA_peak_bytes']) is list and len(r['CUDA_peak_bytes']) == 2 and all(type(v) is int and 0 <= v <= p['profile'][k]
        for v, k in zip(r['CUDA_peak_bytes'], ('cuda_allocated_bytes', 'cuda_reserved_bytes'))), 'CUDA memory caps')
    require(sha(rp) == report_sha and all(sha(f) == h for f, h in p['sources'].items()), 'unchanged source/report closure')
    require_output_fit()
    return dict(schema=NAME+'.consume', status='saved12_common_cap_engineering_verified', prepared=r['prepared'], prepare_sha256=prepare_sha,
        report_sha256=report_sha, denominator=12, request_slots_per_arm=144, total_request_denominator=1728,
        arms=summaries, protocol_sha256=PROTOCOL_SHA, shape_and_cost_proof=proof(),
        completion_sha256=sha(output/'completion.json'), completion=completion, same_cap_vector=True, **FLAGS)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('prepare'); q.add_argument('--output', required=True)
    sub.add_parser('proof')
    for cmd in ('verify', 'run', 'consume'):
        q = sub.add_parser(cmd); q.add_argument('--prepared', required=True); q.add_argument('--prepare-sha', required=True)
        if cmd in ('run', 'consume'): q.add_argument('--output', required=True)
        if cmd == 'run': q.add_argument('--authorization', required=True)
        if cmd == 'consume': q.add_argument('--report-sha', required=True); q.add_argument('--receipt', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'proof': out = proof()
    elif a.cmd == 'prepare': out = prepare(a.output)
    elif a.cmd == 'verify': out = dict(verified=True, schema=verify(a.prepared, a.prepare_sha)['schema'])
    elif a.cmd == 'run': out = run(a.prepared, a.prepare_sha, a.output, authorization=a.authorization)
    else: out = write_new(a.receipt, consume(a.output, a.report_sha, a.prepared, a.prepare_sha))
    print(canonical(out).decode())


if __name__ == '__main__': main()
