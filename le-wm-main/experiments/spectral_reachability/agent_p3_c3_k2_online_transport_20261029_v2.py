"""K2 six-slot online transport adaptation. NOT a standalone HOST runner.

Uses frozen compact CAS and bounded routing; no monkeypatch or old schema change.
A complete K2 source seal, HOST loader, saved reader and proof remain required.
"""
from __future__ import annotations
from copy import deepcopy
from collections import Counter
import os
import shutil
import time
from experiments.spectral_reachability import agent_p3_c3_k2_host_engine_20261029_v1 as engine
from experiments.spectral_reachability import agent_p3_c3_compact_durable_transport_20261029_v1 as d
from experiments.spectral_reachability import agent_p3_c3_k2_common_cap_20261029_v1 as k2

original, codec, routing, native = d.original, d.codec, d.routing, d.native
need, canonical = d.need, d.canonical
NAME = 'agent_p3_c3_k2_online_transport_20261029_v2'
TOTAL, RESERVE, FLOOR = d.TOTAL, d.RESERVE, d.FLOOR
SCOPE_HEADER, SCOPE_COUNT, CALLS = d.SCOPE_HEADER, d.SCOPE_COUNT, d.CALLS
REPLAY_REQUEST = d.REPLAY_REQUEST
FLAGS = d.FLAGS
fsync_dir = d.fsync_dir
Headers, OwnerEncoder = d.Headers, d.OwnerEncoder
require = original.require


def count_envelope():
    """Compare finite K2 resource totals, not a byte-format proof."""
    old_rows, new_rows = original.roster(), k2.roster()
    need(len(old_rows) == 12 and len(new_rows) == 6, 'fixed roster cardinalities')
    old_totals = {a: sum(original.caps(row['K'])[a] for row in old_rows) for a in original.core.AXES}
    new_totals = {a: sum(k2.caps()[a] for _ in new_rows) for a in original.core.AXES}
    need(all(new_totals[a] <= old_totals[a] for a in original.core.AXES), 'all resource axes within old aggregate')
    need(all(k2.caps()[a] <= original.caps(17)[a] for a in original.core.AXES), 'each K2 slot within old K17')
    return dict(old_totals=old_totals, new_totals=new_totals, old_slots=12, new_slots=6,
                request_denominator=864, serialized_byte_fit_proved=False)

class K2Transport(d.Transport):
    def preserve_failure(self, rows, error):
        """Reserved complete literal prefix status, NEVER a successful receipt.

        No cleanup/refund/repair; if an I/O failure also prevents this write,
        existing pack/index/journal bytes remain exactly where they stopped.
        """
        need(type(rows) is list and len(rows) == 6, 'complete failure denominator')
        self.poisoned = True
        value = dict(schema=NAME+'.incomplete_failure', rows=rows,
            error=dict(type=type(error).__name__, message=str(error)),
            committed_journal_events=len(self.journal), full_transport_verified=False, **FLAGS)
        raw = canonical(value); need(len(raw) <= RESERVE, 'complete failure bytes within reserved512MiB')
        # Actual physical size, including a partially written/unindexed suffix,
        # not just the successful in-memory accounting offsets.
        self.disk.checkpoint(); self.journal.sync()
        actual = sum(p.stat().st_size for p in self.directory.iterdir() if p.is_file())
        actual += sum(p.stat().st_size for p in self.disk.directory.iterdir() if p.is_file())
        # Reserve the external prepare and potential receipt maxima even on
        # failure; extra_bytes may also contain internal files, so do not add
        # it wholesale to an already physical census.
        external_reservation = 32*1024**2
        need(actual+len(raw)+external_reservation <= TOTAL and shutil.disk_usage(self.directory).free >= FLOOR+len(raw),
             'failure reserve shared total and floor')
        with (self.directory/'failure.json').open('xb') as f:
            need(f.write(raw) == len(raw), 'whole failure status write')
            f.flush(); os.fsync(f.fileno())
        fsync_dir(self.directory)
        return dict(path=str(self.directory/'failure.json'), bytes=len(raw), sha256=codec.h(raw).hex())


class OnlineStore(d.OnlineStore):
    def __init__(self, directory, store, catalog, tasks):
        need(catalog.store is store and len(tasks) == 2, 'one sealed CAS and two known tasks')
        self.store, self.catalog, self.tasks = store, catalog, tasks
        self.native = native.NativeCatalog(store, catalog)
        self.routing = routing.Routing(store, catalog)
        self.headers = Headers(store, catalog)
        self.encoder = OwnerEncoder(store, catalog)
        self.transport = K2Transport(directory, store)
        self.states = {}; self.slot = None; self.active = None
        self.progress = {}; self.owner_events = Counter()
        self.initial_objects = set(store.objects)

    def resource(self, payload, slot):
        entry = payload['entry']; scope = entry['scope']
        need(scope['mode'] in original.MODES and scope['K'] == 2 and
             scope['phase'] in ('signal', 'schedule', 'program', 'verification'), 'finite original scope values')
        if scope['phase'] == 'verification': need(scope['replay_mode'] in ('whole', 'segmented'), 'original verification mode')
        shape = entry['shape']; op = entry['resource']
        expected_shapes = dict(predict=[1,3,192], action_encoder=[1,3,10],
                               predictor=[1,3,192], pred_proj=[3,192])
        if op in expected_shapes: need(shape == expected_shapes[op] and entry['directions'] is None, 'native batch-one full3 resource shape')
        elif op == 'vjp': need(shape is None and entry['directions'] == 1, 'one actual VJP direction')
        else: need(op in original.core.SCALARS and shape is None and entry['directions'] is None, 'finite native scalar resource')
        k = self.headers.put_header('scope', scope, SCOPE_HEADER, SCOPE_COUNT)
        self.native.bind('scope', scope, dict(slot=slot), k)
        need(entry['resource'] in codec.OPERATIONS and
             entry['delta'] == original.core.delta(entry['resource'], entry['shape'], entry['directions']),
             'exact original delta template, no arbitrary27-vector')
        self.headers.put_header('delta', entry['delta'], routing.AXES, len(codec.OPERATIONS))

    def event(self, event):
        need(not self.transport.poisoned, 'online prefix already failed')
        try:
            if event['kind'] == 'arm_attempt':
                identity = event['identity']
                need(type(identity.get('slot')) is int and 0 <= identity['slot'] < 6
                     and identity == k2.roster()[identity['slot']], 'fixed K2 slot identity')
                self.slot = event['identity']['slot']; self.state(self.slot)
            if event['kind'] != 'arm_event':
                if event['kind'] in ('arm_attempt', 'arm_return'):
                    need(len(canonical(event)) <= 4096, 'finite top row JSON bytes')
                elif event['kind'] == 'denominator':
                    need(len(event['rows']) == 6 and len(canonical(event)) <= 6*4096, 'fixed six top rows')
                else:
                    need(event['kind'] in codec.AUX and all(self.catalog.contains('static', event[n])
                        for n in codec.AUX[event['kind']]), 'fixed source/model/precision values must be in static seal')
                self.transport.append(event, encoder=self.encoder); return
            slot = event['slot']; need(type(slot) is int and 0 <= slot < 6 and slot == self.slot, 'original ordered current K2 slot')
            io = event['payload']; codec.exact(io, ('kind', 'call', 'payload'))
            need(io['kind'] == 'io', 'actual NativeTree IO wrapper')
            payload = io['payload']; cid = io['call']; kind = payload['kind']
            s = self.state(slot); root = self.tasks[k2.roster()[slot]['root']]
            if kind in codec.RESOURCES:
                self.resource(payload, slot)
            if kind == 'native_call_attempt':
                need(self.active is None and type(cid) is int and cid == s['calls'], 'one exact next native invocation')
                self.task(root, payload['task'], dict(slot=slot, call=cid))
                spec = routing.SIGNAL_REQUEST if payload['method'] == 'signal' else REPLAY_REQUEST
                req_key = self.headers.put_header('request', payload['request'], spec, CALLS)
                self.catalog.bind('online_aux', payload['request'], req_key)
                self.active = dict(slot=slot, call=cid, start=len(self.store.events), values=[])
            if cid is not None:
                need(self.active is not None and (slot, cid) == (self.active['slot'], self.active['call']), 'contiguous active native invocation')
                self.active['values'].append(deepcopy(payload))
            else: need(self.active is None, 'no interleaved auxiliary events in native invocation')
            closing = kind in ('native_call_return', 'native_call_failure')
            if closing:
                call = self.active['values']; complete = original.old.audit_native(call)
                need(complete is (kind == 'native_call_return'), 'actual native closure/disposition')
                origin = dict(slot=slot, call=cid)
                if complete:
                    task = call[0]['task']; result = call[-1]['result']
                    if call[0]['method'] == 'signal':
                        for field in ('nominal', 'jacobian'): self.native.bind('native_literal', result[field], origin)
                        for i, e in enumerate(call):
                            if e['kind'] == 'boundary' and e['phase'] == 'nominal':
                                context = dict(task['root_context'],
                                    latent_history=original.signal.unblob(e['arrays']['latent'])[0].tolist(),
                                    action_history=original.signal.unblob(e['arrays']['history'])[0].tolist())
                                self.context(task, context, dict(origin, boundary=i))
                    else:
                        for i, context in enumerate(result): self.context(task, context, dict(origin, boundary=i))
                    spec = routing.RESPONSE if call[0]['method'] == 'signal' else routing.seq(routing.ref('context'), 9)
                    result_key = self.headers.put_header('result', result, spec, CALLS)
                    self.catalog.bind('online_aux', result, result_key)
                else: self.native.failed.add((slot, cid))
            if kind not in native.NATIVE_KINDS: self.aux(payload, slot)
            index = len(self.store.events)
            self.transport.append(event, encoder=self.encoder)
            if kind in ('resource_return', 'resource_failure', 'resource_blocked'): s['finals'].append(index)
            if closing:
                call = self.active['values']; indices = list(range(self.active['start'], index+1))
                view = self.store.add_view('native', indices)
                self.native.bind('transcript', call, dict(slot=slot, call=cid), view)
                self.native.native[slot, cid] = dict(typed_indices=indices)
                if kind == 'native_call_return':
                    capsule = self.store.add_view('capsule', indices); p = native.pointer(slot, cid, call)
                    self.store.capsules[p['path']] = dict(schema=codec.NAME+'.virtual_capsule', path=p['path'],
                        materialized=False, object_sha256=capsule.hex(), slot=slot, call=cid,
                        expanded_canonical_sha256=p['sha256'], expanded_canonical_bytes=len(canonical(call)))
                self.active = None; s['calls'] += 1
        except BaseException:
            self.transport.poisoned = True
            raise


def execute(p, factory, store, *, permit_ram=False, check=lambda: None, on_start=lambda rows: None):
    rows = [dict(**x, status='not_requested', result=None) for x in k2.roster()]; failed = False
    on_start(rows)  # preserve mutated twelve-slot prefix even if the journal fails.
    store.event(dict(kind='denominator', rows=deepcopy(rows), request_slots_per_arm=144))
    for row, identity in zip(rows, k2.roster(), strict=True):
        if failed: row['status'] = 'blocked_after_runtime_failure'; continue
        slot = row['slot']; start = time.monotonic(); store.event(dict(kind='arm_attempt', identity=identity))
        sink = type('Sink', (), {'event': lambda self, e: store.event(dict(kind='arm_event', slot=slot, payload=deepcopy(e)))})()
        def guard():
            check(); require(time.monotonic()-start <= p['profile']['slot_wall_seconds'], 'slot900 wall cap')
        try:
            guard()
            r = engine.run_tree(p['tasks'][row['root']], factory, sink, mode=row['mode'], K=row['K'], permit_ram=permit_ram,
                kind='RAM_FIXTURE' if permit_ram else 'HOST_CUDA', authorization=None if permit_ram else engine.AUTH, guard=guard,
                capsule=lambda cid, call: store.capsule(slot, cid, call))
            ref = store.result(slot, r); row.update(status=r['status'], result=ref, wall_seconds=time.monotonic()-start)
            failed = bool(r['errors']); guard()
        except Exception as exc:
            row.update(status='outer_runtime_failure', error=dict(type=type(exc).__name__, message=str(exc))); failed = True
        store.event(dict(kind='arm_return', row=deepcopy(row)))
    return rows


def validate_rows(p, rows, events, result_reader, capsule_reader, *, permit_fixture=False):
    require(len(rows) == 6 and [{k: r[k] for k in ('slot', 'root', 'mode', 'K')} for r in rows] == k2.roster(), 'complete ordered12 denominator')
    top = {'denominator', 'arm_attempt', 'arm_return', 'source_before', 'source_after', 'model_before', 'model_after', 'precision_before', 'precision_after'}
    require(all(e['kind'] in top | {'arm_event'} for e in events), 'unknown top event')
    require(all(type(e['slot']) is int and 0 <= e['slot'] < 6 for e in events if e['kind'] == 'arm_event'), 'orphan slot')
    initial = [dict(**x, status='not_requested', result=None) for x in k2.roster()]
    require([e for e in events if e['kind'] == 'denominator'] == [dict(kind='denominator', rows=initial, request_slots_per_arm=144)], 'original full denominator')
    summaries = []; failed = False; position = -1
    for row, identity in zip(rows, k2.roster(), strict=True):
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
        summary = engine.validate_tree(p['tasks'][row['root']], r, es, permit_fixture=permit_fixture,
            capsule_reader=lambda cid, call, slot=row['slot']: capsule_reader(slot, cid, call))
        summaries.append(dict(slot=row['slot'], root=row['root'], **summary)); failed = bool(r['errors'])
    require(len([e for e in events if e['kind'] == 'arm_attempt']) == len(summaries) == len([e for e in events if e['kind'] == 'arm_return']), 'no hidden or missing slot dispatch')
    return summaries




