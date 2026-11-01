"""Lossless durable compact transport. No HOST runner or production commands.

append() is online *after* the caller has routed the payload. import_settled()
is explicitly offline: it does not turn NativeCatalog's RAM holes into an
online owner router. All writes are exclusive and all failed bytes retained.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import struct
import time

from experiments.spectral_reachability import agent_p3_c3_compact_native_catalog_20261029_v1 as native

routing, codec, original = native.routing, native.codec, native.original
need, canonical = codec.need, codec.canonical
NAME = 'agent_p3_c3_compact_durable_transport_20261029_v1'
TOTAL, RESERVE, FLOOR = 4*1024**3, 512*1024**2, 200*1024**2
EVENTS, CALLS, CONTEXTS = 2489928, 4884, 6912
ROOTS = CONTEXTS+12+64
MANIFEST_CAP = 16*1024**2
COMPLETION_CAP = 64*1024
METADATA_CAP = codec.STATIC
# Native headers have finite schemas, not arbitrary generic metadata buckets.
CONTEXT_HEADER = routing.rec(layout=routing.ref('static'), context_length=routing.ref('static'),
    latent_dim=routing.ref('static'), action_dim=routing.ref('static'),
    latent_history=routing.arr(3, 192), action_history=routing.arr(2, 10))
TASK_HEADER = routing.rec(binding=routing.BIND, root_context=routing.ref('context'),
    goal=routing.ref('static'), projection=routing.ref('static'),
    action_mean=routing.ref('static'), action_std=routing.ref('static'))
REPLAY_REQUEST = routing.rec(parent_context=routing.ref('context'),
    actions=routing.variant(*(routing.arr(i, 2) for i in range(5, 46, 5))), segmented=routing.B)
SCOPE_HEADER = routing.variant(
    routing.rec(mode=routing.S, K=routing.U, phase=routing.S, parent_context_sha256=routing.H),
    routing.rec(mode=routing.S, K=routing.U, phase=routing.S, replay_mode=routing.S))
CALL_HEADER_BOUND = (routing.bound(TASK_HEADER)+routing.bound(routing.SIGNAL_REQUEST)+
    routing.bound(REPLAY_REQUEST)+routing.bound(routing.RESPONSE)+
    routing.bound(routing.seq(routing.ref('context'), 9))+routing.bound(routing.S))
SCOPE_COUNT = 12*(3*17+2)
HEADER_PARTS = dict(native_task_request_result=CALLS*CALL_HEADER_BOUND,
    # Arrays/payload headers are already charged in the original f64/header
    # axes. Conservatively charge their entire context schema again here.
    complete_context_records=CONTEXTS*routing.bound(CONTEXT_HEADER),
    exact_scopes=SCOPE_COUNT*routing.bound(SCOPE_HEADER),
    delta_templates=len(codec.OPERATIONS)*routing.bound(routing.AXES),
    inclusive_maps=12*routing.bound(routing.rec(**{op:routing.U for op in codec.OPERATIONS})),
    # Top-level fixed12 rows (including failure/null) and counters. Exact entry
    # byte bounds are checked independently, not fixture extrapolation.
    # At most12 initial +12 returned +12 settled rows. For generic JSON
    # payloads128*canonical_bytes bounds raw/type/container/index overhead:
    # each primitive/index <=57+string bytes, each dict member<=34 and
    # container header/index<=53, each syntax node costs at least one byte.
    fixed_top_rows=36*4096*128,
    refusal_metadata=12*(routing.bound(routing.NODE)+2*routing.bound(routing.AXES)+256),
    execution_envelope=64*1024)
PINS = dict(native.SOURCE_PINS)
PINS[native.__file__] = 'bb6ca1839141fb549782543a05efb7de86e60826cd2076b4a6d562dc325f07e7'
FLAGS = dict(production_ready=False, full_C3_demonstrated=False,
             neural_executed=False, physical_release=False, original_R_equivalence=False)


def sources():
    root = Path(__file__).resolve().parents[2]
    paths = (Path(__file__).resolve(), root/'tests'/('test_'+NAME+'.py'), root/'innerdocs'/(NAME+'.md'))
    return {**PINS, **{str(p): codec.file_sha(p) for p in paths}}


def proof():
    """Transport additions, not an invented proof of unimplemented routing.

    Journal104 and every W/index are ALREADY in the frozen base census.
    We do not duplicate event IDs, offsets, view indices or source capsules in
    JSON. All integer sizes below are independently enforced at commit/read.
    The base already includes32MiB static bytes; metadata below conservatively
    charges another32MiB, avoiding an assumption about their overlap.
    """
    base = codec.proof()['live_bytes_upper']
    # Canonical escaped JSON bytes measured BEFORE acceptance, not character
    # counts. Each finite entry is hard capped independent of fixture values.
    manifest_entries = ROOTS*256+CALLS*1024+64*1024
    need(manifest_entries <= MANIFEST_CAP, 'finite manifest entry proof')
    additions = dict(static_json_duplicate=METADATA_CAP,
                     manifest=MANIFEST_CAP, completion=COMPLETION_CAP, **HEADER_PARTS)
    live = base+sum(additions.values())
    return dict(schema=NAME+'.transport_budget', base_conditional_live_bytes=base,
        additions=additions, manifest_entries_bound=manifest_entries,
        max_roots=ROOTS, max_capsules=CALLS, max_events=EVENTS,
        extra_event_index_bytes=0, extra_view_index_bytes=0, npy_base64_overhead=0,
        transport_live_upper=live, live_allowance=TOTAL-RESERVE,
        remaining_live_bytes=(TOTAL-RESERVE)-live,
        total_including_failure_reserve=live+RESERVE, output_cap=TOTAL,
        failure_reserve=RESERVE, floor=FLOOR, whole_wall_seconds=7200,
        slot_wall_seconds=900, gpu_allocated_bytes=3*1024**3,
        gpu_reserved_bytes=4*1024**3, transport_additions_fit=live <= TOTAL-RESERVE,
        native_header_per_call_bound=CALL_HEADER_BOUND, context_count_bound=CONTEXTS,
        scope_count_bound=SCOPE_COUNT, header_schema_fit=live <= TOTAL-RESERVE,
        online_aux_owner_dispatch_before_append=True, finite_native_headers_enforced=True,
        complete_transport_schema_fit_proved=live <= TOTAL-RESERVE,
        complete_production_fit_proved=False,
        missing=['HOST model/class/precision before-after wrapper and source authority build'],
        **FLAGS)


def regular(path, cap):
    p = Path(path).absolute()
    need(p.resolve() == p and not p.is_symlink() and p.is_file()
         and p.stat().st_size <= cap, 'bounded exact regular file')
    return p


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def read_json(path, cap):
    raw = regular(path, cap).read_bytes()
    value = json.loads(raw, object_pairs_hook=codec.unique,
                       parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    need(canonical(value) == raw, 'exact canonical JSON bytes')
    return value


def key(value):
    need(type(value) is str and len(value) == 64 and
         all(c in '0123456789abcdef' for c in value), 'literal canonical SHA256')
    return bytes.fromhex(value)


class Journal(Sequence):
    """Fixed104-byte original-order chain, demand read; no duplicate key file."""
    def __init__(self, path, *, readonly=False):
        self.file = Path(path).open('rb' if readonly else 'x+b')
        self.readonly = readonly
        self.file.seek(0, os.SEEK_END)
        size = self.file.tell()
        need(size % 104 == 0 and size//104 <= EVENTS, 'complete bounded journal records')
        self.count = size//104
    def __len__(self): return self.count
    def __getitem__(self, i):
        if isinstance(i, slice): return [self[j] for j in range(*i.indices(self.count))]
        if type(i) is not int: raise TypeError('literal journal index')
        if i < 0: i += self.count
        if not 0 <= i < self.count: raise IndexError(i)
        self.file.seek(i*104); raw = self.file.read(104)
        need(len(raw) == 104, 'complete journal read')
        return raw
    def append(self, raw):
        need(not self.readonly and type(raw) is bytes and len(raw) == 104
             and self.count < EVENTS, 'writable complete bounded journal append')
        i, previous, ref, tail = codec.JOURNAL.unpack(raw)
        expected = codec.ZERO if not self.count else self[-1][72:]
        need(i == self.count and previous == expected and
             tail == codec.h(codec.CHAIN+raw[:72]), 'original-order hashchain before append')
        self.file.seek(0, os.SEEK_END)
        need(self.file.write(raw) == 104, 'whole journal write')
        self.count += 1
        self.sync()
    def sync(self): self.file.flush(); os.fsync(self.file.fileno())
    def close(self): self.file.close()


class GuardedCAS(routing.DiskCAS):
    """Share one allowance with journal/metadata, not independent4GiB caps."""
    def __init__(self, directory, guard):
        self.guard = guard
        super().__init__(directory, budget=codec.LIVE-codec.STATIC, floor=FLOOR)
    def __setitem__(self, k, raw):
        if k not in self.positions: self.guard(len(raw)+48)
        super().__setitem__(k, raw)


class Transport:
    """Explicit codec mapping/journal boundary; never patches old globals.

    A caller may register bounded routed objects and then append an event.
    CAS checkpoint always precedes the journal commit. A crash can leave CAS
    suffix objects; only the complete manifest is consumable as a settled run.
    The incomplete directory is never deleted or silently 'repaired'.
    """
    def __init__(self, directory, store):
        p = Path(directory).absolute()
        need(p.resolve() == p and not p.exists() and not p.is_symlink(), 'new exclusive transport directory')
        need(not store.events and not store.journal, 'attach before first event; import_settled is separate')
        need(all(codec.file_sha(p) == sha for p, sha in PINS.items()), 'frozen direct sources')
        need(shutil.disk_usage(p.parent).free >= TOTAL+FLOOR, 'whole allowance plus disk floor')
        p.mkdir(); fsync_dir(p.parent)
        self.directory, self.store = p, store
        self.started = time.monotonic(); self.extra_bytes = 0; self.committed = False
        self.poisoned = False
        self.journal = Journal(p/'journal.bin')
        self.disk = GuardedCAS(p/'cas', self.guard)
        try:
            for k, raw in store.objects.items(): self.disk[k] = raw
            need(self.disk.offset+48*len(self.disk) == store.nodes_bytes, 'same actual object/index census')
            store.objects = self.disk; store.journal = self.journal
            self.disk.checkpoint(); fsync_dir(p)
        except BaseException:
            self.close(); raise

    def used(self):
        return self.disk.offset+48*len(self.disk)+104*len(self.journal)+self.extra_bytes

    def guard(self, additional=0):
        # GuardedCAS calls this during construction, before self.disk exists.
        disk = getattr(self, 'disk', None)
        used = (disk.offset+48*len(disk) if disk is not None else 0)
        used += 104*len(self.journal)+self.extra_bytes
        need(used+additional <= TOTAL-RESERVE, 'shared live byte budget before write')
        need(shutil.disk_usage(self.directory).free >= FLOOR+additional, 'shared disk floor before write')
        need(time.monotonic()-self.started <= 7200, 'whole transport7200s wall')

    def append(self, event, *, encoder=None):
        need(not self.committed and not self.poisoned, 'open unpoisoned transport')
        try:
            enc = self.store if encoder is None else encoder
            need(enc is self.store or getattr(enc, 'store', None) is self.store,
                 'encoder must share exact explicit CAS')
            raw, _ = enc._encode_event(event)
            ref = self.store.intern(raw)
            need(canonical(self.store.get(ref)) == canonical(event), 'whole event literal roundtrip')
            self.guard(104); self.disk.checkpoint()  # durable dependencies BEFORE durable root
            i = len(self.journal)
            need(i == len(self.store.events), 'no unresolved positions or late events')
            stem = struct.pack('<Q32s32s', i, self.store.tail, ref)
            tail = codec.h(codec.CHAIN+stem)
            self.journal.append(stem+tail)
            self.store.events.append(ref); self.store.membership.append({'journal'})
            self.store.tail = tail
            kind = codec.Codec.peel(event)[1]['kind']
            self.store.kind_counts[kind] += 1
            self.store.row_max[kind] = max(self.store.row_max.get(kind, 0), len(raw))
            return ref
        except BaseException:
            self.poisoned = True
            raise

    def _write(self, name, value, cap):
        raw = canonical(value); need(len(raw) <= cap, 'finite metadata cap before write')
        self.guard(len(raw))
        with (self.directory/name).open('xb') as f:
            need(f.write(raw) == len(raw), 'whole exclusive metadata write')
            f.flush(); os.fsync(f.fileno())
        self.extra_bytes += len(raw)
        return dict(bytes=len(raw), sha256=codec.h(raw).hex())

    def commit(self):
        need(not self.committed and not self.poisoned, 'complete unpoisoned transport only')
        need(len(self.store.events) == len(self.journal), 'no RAM holes at commit')
        roots = {n: k.hex() for n, k in self.store.roots.items()}
        need(len(roots) <= ROOTS and all(len(canonical({n: v})) <= 256 for n, v in roots.items()),
             'finite root names/JSON entries')
        capsules = self.store.capsules
        need(len(capsules) <= CALLS and all(len(canonical({n: v})) <= 1024 for n, v in capsules.items()),
             'finite complete capsule descriptors')
        self.disk.checkpoint(); self.journal.sync()
        static = self._write('static.json', dict(fields=self.store.fields, static=self.store.static), METADATA_CAP)
        cas = self.disk.finish()
        manifest = dict(schema=NAME+'.fixture_transport', sources=sources(),
            metadata=static, cas=cas, journal=dict(events=len(self.journal),
                bytes=104*len(self.journal), sha256=codec.file_sha(self.directory/'journal.bin'),
                tail=self.store.tail.hex()), roots=roots, capsules=capsules,
            inference='NONE_fixture_only', **FLAGS)
        need(len(canonical({k: v for k, v in manifest.items() if k not in ('roots', 'capsules')}))
             <= 64*1024, 'finite manifest fixed portion')
        binding = self._write('manifest.json', manifest, MANIFEST_CAP)
        fsync_dir(self.directory)
        # Sampling AFTER manifest file+directory fsync, not before report write.
        elapsed = time.monotonic()-self.started
        need(elapsed <= 7200, 'post-manifest-fsync whole wall')
        completion = dict(schema=NAME+'.completion', manifest_sha256=binding['sha256'],
            boundary='manifest_file_and_directory_fsync_returned',
            excludes='completion.json own write/fsync and process shutdown',
            wall_seconds_through_manifest_fsync=elapsed, journal_tail=self.store.tail.hex(),
            artifacts_bytes_before_completion=self.used(), **FLAGS)
        self._write('completion.json', completion, COMPLETION_CAP)
        fsync_dir(self.directory); self.guard()
        self.committed = True
        return dict(path=str(self.directory), manifest_sha256=binding['sha256'],
                    artifacts_bytes=self.used(), **FLAGS)

    def close(self):
        if hasattr(self, 'disk'): self.disk.close()
        if hasattr(self, 'journal'): self.journal.close()

    def preserve_failure(self, rows, error):
        """Reserved complete literal prefix status, NEVER a successful receipt.

        No cleanup/refund/repair; if an I/O failure also prevents this write,
        existing pack/index/journal bytes remain exactly where they stopped.
        """
        need(type(rows) is list and len(rows) == 12, 'complete failure denominator')
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
        need(actual+len(raw) <= TOTAL and shutil.disk_usage(self.directory).free >= FLOOR+len(raw),
             'failure reserve shared total and floor')
        with (self.directory/'failure.json').open('xb') as f:
            need(f.write(raw) == len(raw), 'whole failure status write')
            f.flush(); os.fsync(f.fileno())
        fsync_dir(self.directory)
        return dict(path=str(self.directory/'failure.json'), bytes=len(raw), sha256=codec.h(raw).hex())


class OwnerEncoder(native.CatalogEncoder):
    KINDS = (*native.CatalogEncoder.KINDS, 'online_aux')


class Headers(routing.Routing):
    """Dedicated finite headers; no changes to frozen owner limits/globals."""
    def __init__(self, store, catalog):
        super().__init__(store, catalog)
        self.counts = Counter(); self.values = set(); self.new_bytes = 0
    def _intern(self, raw, owner):
        k = codec.h(codec.DOMAIN+raw)
        if k not in self.store.objects:
            need(self.new_bytes+len(raw)+48 <= sum(HEADER_PARTS.values()), 'complete finite header allocation allowance')
            self.store.intern(raw); self.new_bytes += len(raw)+48
        return k
    def put_header(self, kind, value, spec, count):
        identity = (kind, codec.semantic(value))
        if identity not in self.values:
            need(self.counts[kind] < count, 'finite unique header count: '+kind)
            self.values.add(identity); self.counts[kind] += 1
        return self.encode(value, spec, ('header', 0, 0))


class OnlineStore:
    """Real execute() Store ABI, online routing before every durable append.

    No deferred slot journal or post-hoc event copy. One call buffer supports
    original audit_native; each event is committed immediately. Source capsule
    publication is possible only after its actual return has been audited and
    committed. This wrapper exposes injected RAM execution, not a HOST loader.
    """
    def __init__(self, directory, store, catalog, tasks):
        need(catalog.store is store and len(tasks) == 2, 'one sealed CAS and two known tasks')
        self.store, self.catalog, self.tasks = store, catalog, tasks
        self.native = native.NativeCatalog(store, catalog)
        self.routing = routing.Routing(store, catalog)
        self.headers = Headers(store, catalog)
        self.encoder = OwnerEncoder(store, catalog)
        self.transport = Transport(directory, store)
        self.states = {}; self.slot = None; self.active = None
        self.progress = {}; self.owner_events = Counter()
        self.initial_objects = set(store.objects)

    def context(self, task, value, origin):
        original.core.context(value)
        need(all(canonical(value[n]) == canonical(task['root_context'][n]) for n in
            ('layout', 'context_length', 'latent_dim', 'action_dim')), 'exact fixed context metadata')
        k = self.headers.put_header('context', value, CONTEXT_HEADER, CONTEXTS)
        self.native.bind('context', value, origin, k)
        self.store.roots['context_'+codec.semantic(value)] = k

    def task(self, root, value, origin):
        original.core.validate_task(value)
        need(self.catalog.contains('context', value['root_context']) and
             canonical(value) == canonical(original.parent.parent_task(root, value['root_context'])),
             'exact root/local task authority before header encoding')
        k = self.headers.put_header('task', value, TASK_HEADER, CALLS)
        self.native.bind('task', value, origin, k)

    def resource(self, payload, slot):
        entry = payload['entry']; scope = entry['scope']
        need(scope['mode'] in original.MODES and scope['K'] in (1, 17) and
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

    def state(self, slot):
        return self.states.setdefault(slot, dict(queues=0, used=0, assigned=[], finals=[], calls=0,
                                               decisions={}, banks=set()))

    def aux(self, payload, slot):
        s = self.state(slot); kind = payload['kind']
        ordinal = s['queues'] if kind == codec.MAP_KIND else max(0, s['queues']-1)
        own = ('parent', slot, ordinal)
        def record(name, value, owner=own):
            return self.routing.record(name, value, owner)
        if kind == codec.MAP_KIND:
            record('mapping', payload['mapping'])
        elif kind == 'queue':
            q = payload['queue']; own = ('parent', slot, s['queues'])
            need(s['queues'] < 17, 'original at most17 parent queues')
            record('node', q['parent'], own)
            requests = q['requests']; need(s['used']+len(requests) <= 144, 'original atomic144 identity allowance')
            s['assigned'] = list(range(s['used'], s['used']+len(requests)))
            for i, req in zip(s['assigned'], requests, strict=True): record('request', req, ('candidate', slot, i))
            # Refused requests retain original job status. Their storage owner
            # uses unassigned identities exactly as frozen Routing.tree does.
            if q['status'] == 'queued': s['used'] += len(requests)
            record('queue', q, own); s['queues'] += 1
            s['decisions'][s['queues']-1] = deepcopy(q)
        elif kind == 'progress_bank':
            need(s['decisions'][ordinal]['status'] == 'queued' and
                 s['decisions'][ordinal]['decision']['extension_type'] == 'progress' and ordinal not in s['banks'],
                 'exclusive original progress branch before bank append')
            s['banks'].add(ordinal)
            record('progress', payload['bank'])
            self.progress.setdefault(slot, []).append((ordinal, deepcopy(payload['bank'])))
        elif kind == 'bank_event':
            value = payload['event']
            k = self.routing.encode(value, routing.BANK_EVENT, own)
            self.catalog.bind('online_aux', value, k)
        elif kind == 'R_item_return':
            value = payload['row']
            if value['index'] == 'zero': record('control', value)
            else:
                i = value['index']; need(type(i) is int and 0 <= i < len(s['assigned']), 'actual assigned R8 child index')
                record('proposal', value, ('candidate', slot, s['assigned'][i]))
        elif kind == 'R_return':
            need(s['decisions'][ordinal]['status'] == 'queued' and
                 s['decisions'][ordinal]['decision']['extension_type'] == 'reconfiguration' and ordinal not in s['banks'],
                 'exclusive original R branch before bank append')
            s['banks'].add(ordinal)
            group = payload['result']; record('bank', group['bank'])
            record('control', group['control'])
            for i, value in enumerate(group['rows']): record('proposal', value, ('candidate', slot, s['assigned'][i]))
            record('group', group)
        elif kind == 'signal_refusal':
            # This can be the unscheduled eighteenth parent: it is not charged
            # against one of the17 successful signal/bank buckets.
            record('node', payload['parent'], ('slot', slot, 0))
            for name in ('minimum', 'remaining'):
                k = self.headers.put_header('refusal_axes', payload[name], routing.AXES, 24)
                self.catalog.bind('online_aux', payload[name], k)
        elif kind not in ('R_attempt', 'R_item_attempt'):
            raise ValueError('unknown online auxiliary dispatch')
        self.owner_events[kind] += 1

    def event(self, event):
        need(not self.transport.poisoned, 'online prefix already failed')
        try:
            if event['kind'] == 'arm_attempt':
                self.slot = event['identity']['slot']; self.state(self.slot)
            if event['kind'] != 'arm_event':
                if event['kind'] in ('arm_attempt', 'arm_return'):
                    need(len(canonical(event)) <= 4096, 'finite top row JSON bytes')
                elif event['kind'] == 'denominator':
                    need(len(event['rows']) == 12 and len(canonical(event)) <= 12*4096, 'fixed twelve top rows')
                else:
                    need(event['kind'] in codec.AUX and all(self.catalog.contains('static', event[n])
                        for n in codec.AUX[event['kind']]), 'fixed source/model/precision values must be in static seal')
                self.transport.append(event, encoder=self.encoder); return
            slot = event['slot']; need(slot == self.slot, 'original ordered current slot')
            io = event['payload']; codec.exact(io, ('kind', 'call', 'payload'))
            need(io['kind'] == 'io', 'actual NativeTree IO wrapper')
            payload = io['payload']; cid = io['call']; kind = payload['kind']
            s = self.state(slot); root = self.tasks[original.roster()[slot]['root']]
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

    def capsule(self, slot, cid, call):
        need(self.active is None, 'source cannot precede real audited native return')
        return self.native.capsule(slot, cid, call)

    def result(self, slot, value):
        need(self.active is None and slot == self.slot, 'settled ordered slot result')
        s = self.state(slot); receipt = value['resource_receipt']
        view = self.store.add_view('ledger', s['finals'])
        self.native.bind('ledger', receipt['events'], dict(slot=slot), view)
        inclusive = receipt['inclusive_operation_wall_ns']
        need(set(inclusive) <= set(codec.OPERATIONS), 'finite inclusive operation names')
        k = self.headers.put_header('inclusive', inclusive, routing.rec(**{op:routing.U for op in inclusive}), 12)
        self.native.bind('inclusive', inclusive, dict(slot=slot), k)
        ref = self.routing.tree(slot, value, progress_banks=self.progress.get(slot, ()))
        self.transport.disk.checkpoint()
        return {k: ref[k] for k in ('slot', 'object_sha256', 'expanded_canonical_sha256')}

    def run_fixture(self, factory):
        """Injected RAM factory only. Caller supplies exact sealed fixture table."""
        p = dict(tasks=self.tasks, profile=original.profile())
        pending = []
        def started(rows): pending.append(rows)
        try:
            rows = original.execute(p, factory, self, permit_ram=True, on_start=started)
        except Exception as exc:
            if pending: self.transport.preserve_failure(pending[0], exc)
            raise
        need(not self.transport.poisoned, 'failed transport cannot masquerade as original algorithm failure')
        before = self.store.nodes_bytes
        self.store.roots['fixture_execution'] = self.encoder.put(dict(prepared=p, rows=rows))
        # New blocked-row literals may consume the unused third row tranche;
        # profile/list/root framing itself remains a small exact finite value.
        need(self.store.nodes_bytes-before <= HEADER_PARTS['execution_envelope']+
             12*4096*128, 'whole saved execution envelope plus final rows')
        return rows, self.transport.commit()


def import_settled(directory, store, *, fixture_execution=None, catalog=None):
    """Offline exact transport of a settled RAM archive; NOT online routing.

    Used only for fixture tests. Every existing object/view is copied once;
    original journal order and typed event IDs are retained, not reserialized.
    """
    need(len(store.events) == len(store.journal) and all(type(k) is bytes for k in store.events),
         'complete settled original order')
    if fixture_execution is not None:
        codec.exact(fixture_execution, ('prepared', 'rows'))
        need(catalog is not None and catalog.store is store, 'exact explicit fixture catalog')
        need(len(fixture_execution['rows']) == 12, 'all twelve settled rows including blocked slots')
        encoder = native.CatalogEncoder(store, catalog)
        store.roots['fixture_execution'] = encoder.put(fixture_execution)
    journal, events = list(store.journal), list(store.events)
    store.events = []; store.journal = []
    writer = Transport(directory, store)
    try:
        previous = codec.ZERO
        for i, (ref, raw) in enumerate(zip(events, journal, strict=True)):
            index, prior, saved, tail = codec.JOURNAL.unpack(raw)
            need(index == i and prior == previous and saved == ref and
                 tail == codec.h(codec.CHAIN+raw[:72]), 'whole imported journal chain')
            writer.guard(104); writer.journal.append(raw); store.events.append(ref)
            previous = tail
        need(previous == store.tail, 'whole imported final tail')
        writer.commit_result = writer.commit()
        return writer
    except BaseException:
        writer.poisoned = True; writer.close(); raise


class Reader:
    """Readonly complete file/chain/view consumer. Does NOT run fullstrict."""
    def __init__(self, directory, expected):
        directory = Path(directory).absolute(); key(expected)
        need(directory.resolve() == directory and not directory.is_symlink(), 'exact transport directory')
        need({p.name for p in directory.iterdir()} ==
             {'cas', 'journal.bin', 'static.json', 'manifest.json', 'completion.json'}, 'whole transport file census')
        manifest_path = regular(directory/'manifest.json', MANIFEST_CAP)
        need(codec.file_sha(manifest_path) == expected, 'actual manifest SHA')
        m = read_json(manifest_path, MANIFEST_CAP)
        codec.exact(m, ('schema', 'sources', 'metadata', 'cas', 'journal', 'roots', 'capsules', 'inference', *FLAGS))
        need(m['schema'] == NAME+'.fixture_transport' and m['sources'] == sources() and
             all(m[k] is False for k in FLAGS) and m['inference'] == 'NONE_fixture_only', 'own fixture ABI and source closure')
        need(all(codec.file_sha(p) == sha for p, sha in m['sources'].items()), 'current frozen direct and own source bytes')
        codec.exact(m['metadata'], ('bytes', 'sha256'))
        static_path = regular(directory/'static.json', METADATA_CAP)
        need(m['metadata'] == dict(bytes=static_path.stat().st_size, sha256=codec.file_sha(static_path)), 'static exact file SHA/size')
        static = read_json(static_path, METADATA_CAP); codec.exact(static, ('fields', 'static'))
        self.store = codec.Codec(static['fields'], static['static'])
        need(static['fields'] == self.store.fields, 'exact sorted unique sealed field table')
        disk = routing.DiskCAS.open_saved(directory/'cas', m['cas'])
        self.disk, self.journal = disk, None
        try:
            self.store.objects = disk; self.store.nodes_bytes = disk.offset+len(disk)*48
            j = m['journal']; codec.exact(j, ('events', 'bytes', 'sha256', 'tail'))
            codec.uint(j['events']); need(j['events'] <= EVENTS and j['bytes'] == 104*j['events'], 'whole journal denominator')
            p = regular(directory/'journal.bin', EVENTS*104)
            need(p.stat().st_size == j['bytes'] and codec.file_sha(p) == j['sha256'], 'journal complete bytes/SHA')
            self.journal = Journal(p, readonly=True); self.store.journal = self.journal
            previous = codec.ZERO; self.native_indices = {}
            for i, raw in enumerate(self.journal):
                index, prior, ref, tail = codec.JOURNAL.unpack(raw)
                need(index == i and prior == previous and tail == codec.h(codec.CHAIN+raw[:72]) and
                     ref in disk and disk[ref][:1] == b'E', 'whole saved typed journal hashchain')
                self.store.events.append(ref); self.store.membership.append({'journal'})
                event_raw = disk[ref]; cursor = codec.Cursor(event_raw)
                cursor.take(1); cursor.number('<B'); flags = cursor.number('<B')
                slot = cursor.number('<B'); cid = cursor.option()
                need(flags in (0, 1, 2, 3) and (slot < 12 if flags & 1 else slot == 255)
                     and (flags & 2 or cid is None), 'exact saved event wrapper prefix')
                if cid is not None:
                    need(flags == 3, 'native call belongs to explicit slot/io wrapper')
                    self.native_indices.setdefault((slot, cid), []).append(i)
                previous = tail
            need(len(self.journal) == j['events'] and previous == key(j['tail']), 'exact final tail/count')
            self.store.tail = previous
            need(type(m['roots']) is dict and len(m['roots']) <= ROOTS and all(
                len(canonical({n: v})) <= 256 for n, v in m['roots'].items()), 'bounded root entries')
            self.store.roots = {n: key(v) for n, v in m['roots'].items()}
            need(type(m['capsules']) is dict and len(m['capsules']) <= CALLS and all(
                len(canonical({n: v})) <= 1024 for n, v in m['capsules'].items()), 'bounded complete capsule entries')
            self.store.capsules = m['capsules']
            # W carries its own original indices; do not write duplicated arrays
            # in manifest or accept fabricated view membership supplied by JSON.
            for k in disk:
                raw = disk[k]
                if raw[:1] != b'W': continue
                c = codec.Cursor(raw); c.take(1); family = c.number('<B')
                need(family < 4, 'finite view family')
                count = c.number('<I'); need(0 < count <= len(self.journal) and len(raw) == 6+8*count, 'bounded view shape before allocation')
                indices = [c.number('<Q') for _ in range(count)]; c.end()
                name = ('slot', 'native', 'capsule', 'ledger')[family]
                need(indices == sorted(set(indices)) and all(i < len(self.journal) for i in indices), 'ordered existing view members')
                need(all(name not in self.store.membership[i] for i in indices), 'nonoverlapping same-family views')
                self.store.views[k] = name, indices
                for i in indices: self.store.membership[i].add(name)
            self.completion = read_json(directory/'completion.json', COMPLETION_CAP)
            w = self.completion
            codec.exact(w, ('schema', 'manifest_sha256', 'boundary', 'excludes', 'wall_seconds_through_manifest_fsync',
                'journal_tail', 'artifacts_bytes_before_completion', *FLAGS))
            need(w['schema'] == NAME+'.completion' and w['manifest_sha256'] == expected and
                 w['boundary'] == 'manifest_file_and_directory_fsync_returned' and
                 w['excludes'] == 'completion.json own write/fsync and process shutdown' and
                 w['journal_tail'] == j['tail'] and all(w[k] is False for k in FLAGS), 'explicit post-manifest completion binding')
            wall = w['wall_seconds_through_manifest_fsync']
            need(type(wall) in (float, int) and 0 <= wall <= 7200, 'post-manifest wall bound')
            sizes = sum(p.stat().st_size for p in directory.iterdir() if p.is_file())+disk.offset+48*len(disk)
            codec.uint(w['artifacts_bytes_before_completion'])
            need(w['artifacts_bytes_before_completion'] == sizes-(directory/'completion.json').stat().st_size
                 and sizes <= TOTAL-RESERVE, 'all physical files shared live census')
            self.bytes, self.manifest = sizes, m
            self.verify_reachability()
        except BaseException:
            self.close(); raise

    def event(self, index): return self.store.event(index)

    def verify_reachability(self):
        """Raw reference walk: no tensor expansion, no hidden CAS suffix.

        The frozen E decoder is used solely as a structural reference parser;
        get records32-byte child references without deserializing their values.
        W indices are linked to the authenticated original-order journal.
        This is transport closure, NOT original signal/beam mathematics.
        """
        visited, active = set(), set()
        def walk(k):
            need(k in self.disk and k not in active, 'existing noncyclic transport reference')
            if k in visited: return
            active.add(k); raw = self.disk[k]; c = codec.Cursor(raw)
            tag = c.take(1); children = []
            if tag == b'M':
                names = set()
                for _ in range(c.number('<H')):
                    code = c.number('<H')
                    need(code < len(self.store.fields) and code not in names, 'unique known field codes')
                    names.add(code); children.append(c.take(32))
            elif tag == b'L':
                count = c.number('<I'); need(len(raw) == 5+32*count, 'exact reference-list length')
                children = [c.take(32) for _ in range(count)]
            elif tag == b'E':
                class References:
                    fields = self.store.fields
                    _field = codec.Codec._field
                    def get(_, child, trail): children.append(child); return None
                codec.Codec._decode_event(References(), c, frozenset())
            elif tag == b'W':
                family = c.number('<B'); count = c.number('<I')
                need(family < 4 and 0 < count <= len(self.journal) and len(raw) == 6+8*count,
                     'bounded exact view reference list')
                indices = [c.number('<Q') for _ in range(count)]
                need(indices == sorted(set(indices)) and all(i < len(self.journal) for i in indices), 'exact view indices')
                children = [self.store.events[i] for i in indices]
            elif tag in (b'N', b'T', b'F', b'I', b'U', b'D', b'S', b'B', b'V'):
                # Original decoder validates literal dtype/shape/finite/-0.
                self.store.get(k); c.pos = len(raw)
            else: raise ValueError('unknown transport tag')
            c.end()
            for child in children: walk(child)
            active.remove(k); visited.add(k)
        roots = list(self.store.events)+list(self.store.roots.values())
        for path, entry in self.store.capsules.items():
            codec.exact(entry, ('schema', 'path', 'materialized', 'slot', 'call', 'object_sha256',
                                'expanded_canonical_sha256', 'expanded_canonical_bytes'))
            need(entry['schema'] == codec.NAME+'.virtual_capsule' and entry['path'] == path and
                 entry['materialized'] is False, 'explicit nonmaterial capsule descriptor')
            codec.uint(entry['slot']); codec.uint(entry['call'])
            need(entry['slot'] < 12 and path == f'/__virtual__/{native.NAME}/s{entry["slot"]}_c{entry["call"]}.json',
                 'exact complete source capsule namespace')
            k = key(entry['object_sha256']); need(k in self.store.views and self.store.views[k][0] == 'capsule', 'real capsule W root')
            roots.append(k)
        for k in roots: walk(k)
        need(visited == set(self.disk), 'no unused or hidden CAS objects')
        return len(visited)

    def capsule(self, slot, cid, call):
        p = native.pointer(slot, cid, call)
        entry = self.store.capsules[p['path']]
        need(entry['materialized'] is False and entry['slot'] == slot and entry['call'] == cid,
             'nonmaterial exact slot/call source capsule')
        k = key(entry['object_sha256']); indices = self.native_indices.get((slot, cid), [])
        need(indices and indices == list(range(indices[0], indices[-1]+1)) and
             self.store.views[k] == ('capsule', indices), 'all and only indexed original native invocation')
        value = self.store.get(k)
        need(codec.semantic(value) == entry['expanded_canonical_sha256'] and
             len(canonical(value)) == entry['expanded_canonical_bytes'] and
             p['sha256'] == entry['expanded_canonical_sha256'], 'whole indexed semantic capsule bytes')
        need(canonical(value) == canonical(call) and value[-1]['kind'] == 'native_call_return',
             'complete original call required, not failure prefix source')
        return p

    def result(self, slot, ref):
        codec.exact(ref, ('slot', 'object_sha256', 'expanded_canonical_sha256'))
        value = self.store.get(self.store.roots[f'slot_{slot:02}'])
        need(type(ref) is dict and ref.get('slot') == slot and
             ref.get('object_sha256') == self.store.roots[f'slot_{slot:02}'].hex() and
             ref.get('expanded_canonical_sha256') == codec.semantic(value), 'whole saved result reference')
        return value

    def summary(self):
        return dict(schema=NAME+'.saved_transport', events=len(self.journal), objects=len(self.disk),
                    roots=len(self.store.roots), capsules=len(self.store.capsules),
                    physical_bytes=self.bytes, **FLAGS)

    def fixture_inputs(self):
        """Literal saved inputs; caller must explicitly opt into RAM validator.

        This accessor executes no original reconstruction and cannot certify
        a HOST run. The archived twelve rows include blocked/unrequested slots.
        """
        value = self.store.get(self.store.roots['fixture_execution'])
        codec.exact(value, ('prepared', 'rows'))
        need(type(value['rows']) is list and len(value['rows']) == 12, 'full saved fixture denominator')
        return value['prepared'], value['rows']

    def reconstruct_fixture(self):
        """Original saved mathematics, hard restricted to RAM_FIXTURE shards."""
        prepared, rows = self.fixture_inputs()
        events = [self.event(i) for i in range(len(self.journal))]
        def result(slot, ref):
            value = self.result(slot, ref)
            need(value['backend_kind'] == 'RAM_FIXTURE', 'fixture reconstruction refuses HOST production shards')
            return value
        summaries = original.validate_rows(prepared, rows, events, result, self.capsule, permit_fixture=True)
        return dict(schema=NAME+'.fixture_saved_reconstruction', denominator=12, requests=1728,
            slots=[dict(slot=s['slot'], status=s['status'], plan_available=s['plan_available'],
                attempted=s['resources']['attempted'], completed=s['resources']['completed']) for s in summaries],
            all_rows=rows, mathematical_implementation='reused original.validate_rows/validate_tree', **FLAGS)

    def close(self):
        if getattr(self, 'journal', None) is not None: self.journal.close()
        if getattr(self, 'disk', None) is not None: self.disk.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('proof', help='integer transport-only census; NOT complete production fit')
    for name, help_text in (
        ('check-transport', 'readonly fixture file/chain/view check; NOT fullstrict'),
        ('fixture-reconstruct', 'original saved reconstruction; hard rejects HOST_CUDA shards')):
        q = sub.add_parser(name, help=help_text)
        q.add_argument('--directory', required=True); q.add_argument('--manifest-sha256', required=True)
    args = p.parse_args()
    if args.command == 'proof': print(canonical(proof()).decode()); return
    reader = Reader(args.directory, args.manifest_sha256)
    try: print(canonical(reader.reconstruct_fixture() if args.command == 'fixture-reconstruct' else reader.summary()).decode())
    finally: reader.close()


if __name__ == '__main__': main()
