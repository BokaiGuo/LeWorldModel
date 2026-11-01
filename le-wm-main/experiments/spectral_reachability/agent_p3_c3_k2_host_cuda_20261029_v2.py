"""Own HOST wrapper for frozen online compact C3; MAIN alone may run CUDA.

Import, proof and saved reconstruction never import torch. RAM injection is a
separate schema, not a way to create HOST receipts. No old run/consume/gate is
called or patched. Complete source discovery is intentionally MAIN-only.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.spectral_reachability import agent_p3_c3_compact_durable_transport_20261029_v1 as durable
from experiments.spectral_reachability import agent_p3_c3_k2_online_transport_20261029_v2 as ktransport
from experiments.spectral_reachability import agent_p3_c3_k2_host_engine_20261029_v1 as engine
from experiments.spectral_reachability import agent_p3_c3_k2_count_bound_20261029_v2 as kbound
k2 = engine.k2

original, codec, routing, native = durable.original, durable.codec, durable.routing, durable.native
core, old, signal = original.core, original.old, original.signal
need, canonical = durable.need, durable.canonical
NAME = 'agent_p3_c3_k2_host_cuda_20261029_v2'
ROOT = original.ROOT
PREPARE_CAP, REPORT_CAP, RECEIPT_CAP = 16*1024**2, 8*1024**2, 16*1024**2
CLASS_CAP, HOST_CAP = 2*1024**2, 64*1024
EXTERNAL_RESERVATION = PREPARE_CAP+RECEIPT_CAP
PINS = {**durable.sources()}
PINS[str(Path(durable.__file__).resolve())] = '5142bbee6d829aa245ea4019ff55c4d050d28c5c6fa9472dfef874bd05c42041'
# The dependency triple is frozen; never compute its supposed authority anew.
PINS[str(ROOT/'tests'/('test_'+durable.NAME+'.py'))] = 'b6ed5749d4b505e5d47bf137e660fd5f5879755c9b8643d4d1572f49f9057bf4'
PINS[str(ROOT/'innerdocs'/(durable.NAME+'.md'))] = 'dbc1a46f9a174e9abce625fea8b0c89d949bc639eb028b6b0521a99ffbee7812'
PINS[str(Path(k2.__file__).resolve())] = '3b32f147af8fa3baed36634461cd83624cba0900ff7b4fbb37c4cb89ae8f5aec'
PINS[str(ROOT/'tests'/('test_'+k2.NAME+'.py'))] = '09c8d9bc16cca5d5b0e5aaf7eb531201f856448d0109ba6e34485276880a1de8'
PINS[str(ROOT/'innerdocs'/(k2.NAME+'.md'))] = 'faadf4d5f1a4ee5a05dc25c8abb13cdbe5b2567bc6f1821e10628bf3b8a38108'
FLAGS = dict(original.FLAGS, production_ready=False)


def own_files():
    return (Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md'))


def profile():
    value = original.profile()
    value['resource_caps'] = {'2': k2.caps()}
    return value


def rules():
    value = original.rules()
    value['inherited_original_protocol_sha256'] = value.pop('protocol_sha256')
    value['roster'] = k2.roster()
    return value


def check_sources(sources):
    need(type(sources) is dict and sources, 'nonempty actual source closure')
    for name, expected in sources.items():
        durable.key(expected)
        # Original authority includes the69MiB weights asset. Stream its SHA;
        # the32MiB limit is OUTPUT static metadata, not input file size.
        need(codec.file_sha(durable.regular(name, 2**63-1)) == expected, 'actual sealed source bytes: '+name)
    need(all(sources.get(n) == h for n, h in PINS.items()), 'all frozen transport dependencies')
    need(all(sources.get(str(p)) == codec.file_sha(p) for p in own_files()), 'own complete triple sources')


def proof():
    """No average compression or fixture extrapolation; enforced finite caps.

    Frozen transport includes static typed pack+index+table<=32MiB, duplicate
    static.json<=32MiB, manifest16MiB and completion64KiB. HOST data is in that
    very same static budget, NOT another free metadata namespace. Archive p
    is an already registered static reference. Top rows use the frozen third
    12-row tranche; their framing uses the existing64KiB envelope. The only
    new physical allocations are report, external prepare and external receipt.
    They are all charged to live space, leaving the failure512MiB intact.
    """
    base = durable.proof()
    counts = kbound.bound()
    structural = routing.proof()
    need(base['complete_transport_schema_fit_proved'] and structural['all_structural_buckets_fit'],
         'frozen typed transport and per-owner bounds')
    need(counts['old_bytes'] == base['base_conditional_live_bytes'], 'exact frozen codec baseline')
    extra = dict(external_prepare=PREPARE_CAP, host_report=REPORT_CAP, external_receipt=RECEIPT_CAP)
    live = base['transport_live_upper']+sum(extra.values())
    return dict(schema=NAME+'.budget', transport=base, k2_conditional_codec=counts,
        structural_schema=structural, additional_physical_files=extra,
        live_bytes_upper=live, failure_reserve=durable.RESERVE,
        total_bytes_upper=live+durable.RESERVE, cap=durable.TOTAL,
        live_remaining=durable.TOTAL-durable.RESERVE-live,
        fits=live <= durable.TOTAL-durable.RESERVE,
        proof_boundary='frozen typed-schema bounds, K2 resource dominance, bounded metadata and shared writer guard; no time or hardware fit prediction',
        source_static_pack_index_table_cap=codec.STATIC,
        source_static_JSON_cap=durable.METADATA_CAP,
        classes_canonical_bytes_cap=CLASS_CAP, host_canonical_bytes_cap=HOST_CAP,
        rows_canonical_bytes_cap=6*4096, report_cap=REPORT_CAP,
        output_support=dict(slots=6, request_identities=864, axes=27),
        wall_seconds=7200, slot_wall_seconds=900, cuda_allocated_bytes=3*1024**3,
        cuda_reserved_bytes=4*1024**3, floor=durable.FLOOR,
        wall_or_GPU_fit_predicted=False, static_acceptance='actual real authority + actual loaded model metadata; before first append',
        **FLAGS)


def build():
    # Actual frozen upstream authority/task discovery, not codec.keys_of of a
    # fixture, saved outcome selection, or old blocked HOST run/consume.
    check_sources({**PINS, **{str(p): codec.file_sha(p) for p in own_files()}})
    engine.check_k2_source()
    prior = original.build()
    sources = dict(prior['sources']); sources.update(PINS)
    sources.update({str(p): codec.file_sha(p) for p in own_files()})
    sources.update({str(Path(x.__file__).resolve()):codec.file_sha(x.__file__) for x in (k2, engine, ktransport, kbound)})
    p = dict(schema=NAME+'.prepare', tasks=prior['tasks'], policy=prior['policy'],
        expected_model_hash=prior['expected_model_hash'], authority=prior['authority'],
        sources=sources, roster=k2.roster(), profile=profile(),
        precision=old.precision_contract(), rules=rules(), k2_protocol=k2.protocol(),
        selection_history='exploratory K2 after K1 queue refusal', byte_proof=proof(), **FLAGS)
    check_sources(p['sources'])
    need(len(canonical(p)) <= PREPARE_CAP and proof()['fits'], 'finite actual authority and fullshape output proof')
    return p


def write_new(path, value, cap):
    p = Path(path).absolute()
    need(p.parent == ROOT/'results' and p.name.startswith(NAME) and p.resolve() == p and not p.exists(), 'exclusive own result')
    raw = canonical(value); need(len(raw) <= cap, 'finite complete metadata')
    need(shutil.disk_usage(p.parent).free >= durable.FLOOR+len(raw), 'metadata disk floor')
    with p.open('xb') as f:
        need(f.write(raw) == len(raw), 'whole metadata write'); f.flush(); os.fsync(f.fileno())
    durable.fsync_dir(p.parent)
    return dict(path=str(p), sha256=codec.file_sha(p))


def prepare(output): return write_new(output, build(), PREPARE_CAP)


def verify(prepared, expected):
    durable.key(expected); path = durable.regular(prepared, PREPARE_CAP)
    need(codec.file_sha(path) == expected, 'external prepare SHA')
    p = durable.read_json(path, PREPARE_CAP)
    need(canonical(p) == canonical(build()) and codec.file_sha(path) == expected, 'complete original authority reconstruction')
    return p


def bank_literals():
    s = original.tree.spectral
    return dict(rule=s.contract(), sources={str(p): core.sha(p) for p in s.PINS},
        numeric_environment=dict(numpy_version=np.__version__, float64_dtype=np.dtype(np.float64).str,
                                 float32_dtype=np.dtype(np.float32).str))


def json_literal(value):
    """Preserve exact canonical JSON while dropping runtime scalar subclasses.

    In particular torch.__version__ can be a str subclass; the frozen typed
    codec deliberately accepts only literal built-in JSON values. This does not
    coerce a non-JSON foreign object or change numeric precision.
    """
    raw = canonical(value)
    result = json.loads(raw, object_pairs_hook=codec.unique)
    need(canonical(result) == raw, 'exact literal JSON boundary')
    return result


def make_seal(p, before, *, fixture=False):
    """Real measurements, fixed schema names only; never discover future IO.

    Install performs the complete typed/static32MiB census. HOST rejects RAM
    schema even when a test supplies plausible-looking model/precision hashes.
    """
    need(p['schema'] == NAME+('.ram_prepare' if fixture else '.prepare'), 'distinct authority schema')
    check_sources(p['sources'])
    validate_before(p, before)
    # Fixed native model hook field names (not data-dependent key discovery).
    fields = {'kind', 'slot', 'root', 'mode', 'K', 'status', 'result', 'wall_seconds', 'error',
        'type', 'message', 'object_sha256', 'expanded_canonical_sha256', 'prepared', 'rows',
        'shape', 'dtype', 'sha256', 'data', 'segmented', 'latent', 'history', 'raw_flat',
        'mean', 'std', 'raw', 'token', 'outgoing', 'value', 'backend_kind'}
    fields.update(x for op in ('action_encoder','predict','predictor','pred_proj','grad')
        for x in (op+':0', op+':1', op+':output'))
    fields.update(('predict:latent','predict:act_emb','grad:output','grad:raw','grad:basis'))
    fields.update(codec.OPERATIONS)  # finite inclusive_operation_wall_ns keys
    for spec in (durable.CONTEXT_HEADER,durable.TASK_HEADER,durable.REPLAY_REQUEST,durable.SCOPE_HEADER,
                 routing.SIGNAL_REQUEST,routing.RESPONSE,routing.AXES):
        fields.update(routing.fields_of(spec))
    small, large = {}, []
    for name, sha in sorted(p['sources'].items()):
        n = Path(name).stat().st_size
        if n <= codec.STATIC: small[name] = sha
        else: large.append(dict(path=name,sha256=sha,bytes=n))
    extra = dict(fields=sorted(fields), prepared=p, before=before, bank=bank_literals(),
        source_manifest_sha256=core.digest(p['sources']), precision=old.precision_contract(),
        large_input_assets=large, all_source_bytes_checked=True,
        execution_backend='RAM_FIXTURE' if fixture else 'HOST_CUDA')
    # Frozen seal_static's per-INPUT-file32MiB condition is not a weights
    # loader contract. Explicitly separate streaming-authenticated large input
    # assets; no missing file or unverified source is dropped from p/extra.
    seal = routing.seal_static(p['tasks'], small, extra, read_sources=True)
    store, catalog, census = routing.install_static(seal)
    need(census['source_bytes_checked'] is True and census['total_static_bytes'] <= codec.STATIC,
         'real source seal and full actual typed static census before native dispatch')
    return seal, store, catalog, census


def validate_before(p, before):
    codec.exact(before, ('model', 'classes', 'precision', 'host'))
    codec.exact(before['model'], ('original_state_dict', 'all_buffers'))
    for h in before['model'].values(): durable.key(h)
    need(before['model']['original_state_dict'] == p['expected_model_hash'], 'original model weights')
    classes = before['classes']
    need(type(classes) is list and classes and len(canonical(classes)) <= CLASS_CAP, 'finite actual installed classes')
    names = []
    for row in classes:
        codec.exact(row, ('name','class_name','path','sha256'))
        need(all(type(v) is str for v in row.values()) and p['sources'].get(row['path']) == row['sha256'], 'presealed installed class source')
        names.append(row['name'])
    need(len(names) == len(set(names)), 'unique complete named module rows')
    need(before['precision'] == old.precision_contract(), 'MATH/noTF32/noautocast')
    h = before['host']
    need(len(canonical(h)) <= HOST_CAP and type(h) is dict and h['device'] == 'cuda:0'
         and type(h['allocation_sum']) is float and h['allocation_sum'] == 4. and h['visible_nvidia_nodes'], 'actual CUDA preflight ABI')


class Guard:
    def __init__(self, started, memory, *, clock=time.monotonic):
        self.started, self.memory, self.clock = started, memory, clock
        self.transport = None
        self.extra_check = lambda: None
    def __call__(self):
        need(0 <= self.clock()-self.started <= 7200, 'whole7200 includes verify/load/serialize/fsync')
        peak = self.memory()
        need(type(peak) in (tuple,list) and len(peak) == 2 and all(type(x) is int and 0 <= x <= cap
            for x, cap in zip(peak, (3*1024**3, 4*1024**3))), 'CUDA allocated3/reserved4GiB guard')
        if self.transport is not None:
            self.transport.guard()
        self.extra_check()
        return list(peak)


def census_summary(c):
    """Lossless reference to full static entries, small fixed report envelope."""
    codec.exact(c,('schema','canonical_bytes','object_and_index_bytes','field_and_metadata_bytes',
        'total_static_bytes','cap','source_bytes_checked','source_entries','production_ready'))
    value = {k:v for k,v in c.items() if k != 'source_entries'}
    value.update(source_entries_count=len(c['source_entries']),source_entries_sha256=codec.semantic(c['source_entries']))
    need(len(canonical(value)) <= 4096, 'bounded actual static census summary')
    return value


def receipt_arm(s):
    """Full ledger is consumed/hashed; receipt does not duplicate all events."""
    v = deepcopy(s); resources = v['resources']; events = resources.pop('events')
    need(type(events) is list, 'complete actual original ledger events')
    resources.update(events_count=len(events),events_semantic_sha256=codec.semantic(events))
    need(len(canonical(v)) <= 16384, 'finite exact per-arm receipt summary')
    return v


def commit_host(online, p, rows, before, after, census, prepared, prepare_sha, guard, *, fixture):
    """Own manifest, no relabeling of a fixture transport or frozen commit()."""
    t = online.transport; store = online.store
    need(not t.poisoned and online.active is None, 'complete online transport')
    need(canonical(before) == canonical(after), 'actual model/class/precision/host unchanged')
    need(len(rows) == 6 and len(canonical(rows)) <= 6*4096, 'all12 rows finite')
    need(all(r['status'] in original.SETTLED and r['result'] is not None for r in rows), 'complete settled12, not partial model plan')
    # p is already in static; row references point to actual routed slot DAGs.
    start = store.nodes_bytes
    store.roots['host_execution'] = online.encoder.put(dict(prepared=p, rows=rows))
    need(store.nodes_bytes-start <= durable.HEADER_PARTS['execution_envelope']+6*4096*128, 'existing row/envelope allowance')
    need(len(canonical(str(Path(prepared).absolute()))) <= 4096, 'bounded literal authority path')
    report = dict(schema=NAME+('.ram_report' if fixture else '.report'),
        status='six_slots_settled_not_admitted', execution_backend='RAM_FIXTURE' if fixture else 'HOST_CUDA',
        prepared=str(Path(prepared).absolute()), prepare_sha256=prepare_sha,
        rows=rows, before=before, after=after, static_census=census_summary(census),
        sources_sha256=core.digest(p['sources']), CUDA_peak_bytes=guard(), errors=[],
        wall_seconds_before_report=guard.clock()-guard.started,
        journal_tail=store.tail.hex(), neural_executed=not fixture, **FLAGS)
    t._write('report.json', report, REPORT_CAP)
    t.disk.checkpoint(); t.journal.sync()
    metadata = t._write('static.json', dict(fields=store.fields, static=store.static), durable.METADATA_CAP)
    roots = {n:k.hex() for n,k in store.roots.items()}
    need(len(roots) <= durable.ROOTS and all(len(canonical({n:v})) <= 256 for n,v in roots.items()), 'finite all roots')
    need(len(store.capsules) <= durable.CALLS and all(len(canonical({n:v})) <= 1024 for n,v in store.capsules.items()), 'finite all capsules')
    m = dict(schema=NAME+('.ram_transport' if fixture else '.transport'),
        sources={**PINS, **{str(f):codec.file_sha(f) for f in own_files()}},
        source_manifest_sha256=core.digest(p['sources']), metadata=metadata, cas=t.disk.finish(),
        journal=dict(events=len(t.journal),bytes=104*len(t.journal),sha256=codec.file_sha(t.directory/'journal.bin'),tail=store.tail.hex()),
        roots=roots, capsules=store.capsules, report=dict(bytes=(t.directory/'report.json').stat().st_size,
            sha256=codec.file_sha(t.directory/'report.json')), **FLAGS)
    need(len(canonical({k:v for k,v in m.items() if k not in ('roots','capsules')})) <= 64*1024, 'manifest fixed64KiB with full sources')
    manifest = t._write('manifest.json', m, durable.MANIFEST_CAP); durable.fsync_dir(t.directory)
    peak = guard()
    completion = dict(schema=NAME+'.completion', manifest_sha256=manifest['sha256'],
        report_sha256=m['report']['sha256'], journal_tail=store.tail.hex(),
        wall_seconds_through_manifest_fsync=guard.clock()-guard.started,
        boundary='manifest_report_files_and_directory_fsync_returned',
        excludes='completion.json own write/fsync and process shutdown',
        CUDA_peak_bytes=peak, artifacts_bytes_before_completion=t.used()-EXTERNAL_RESERVATION, **FLAGS)
    final = t._write('completion.json', completion, durable.COMPLETION_CAP)
    durable.fsync_dir(t.directory); guard(); t.committed = True
    return dict(output=str(t.directory), manifest_sha256=manifest['sha256'],
                report_sha256=m['report']['sha256'], completion_sha256=final['sha256'], **FLAGS)


def drive(p, prepared, prepare_sha, output, before, factory, final_check, guard, *, fixture=False):
    """Explicit injected runtime seam; no production model loader in RAM mode."""
    _, store, catalog, census = make_seal(p, before, fixture=fixture)
    guard()
    class OnlineStore(ktransport.OnlineStore):
        first_transport_error = None
        def event(self, value):
            try: return super().event(value)
            except BaseException as exc:
                if self.first_transport_error is None: self.first_transport_error = exc
                raise
    online = OnlineStore(output, store, catalog, p['tasks'])
    t = online.transport; t.started = guard.started; guard.transport = t
    # Charge the external maxima inside the underlying transport's shared
    # pre-append ledger itself, not just at the occasional Session guard.
    # extra_bytes includes these virtual reservations; completion subtracts
    # them when reporting physical bytes inside the directory.
    t.extra_bytes += EXTERNAL_RESERVATION
    guard()
    rows = [dict(**x,status='blocked_before_model',result=None) for x in k2.roster()]
    def started(value):
        nonlocal rows
        rows = value
    try:
        for e in (dict(kind='source_before',sha256=core.digest(p['sources'])),
                  dict(kind='model_before',hashes=before['model'],classes=before['classes']),
                  dict(kind='precision_before',value=before['precision'])): online.event(e)
        rows = ktransport.execute(p, factory, online, permit_ram=fixture, check=guard, on_start=started)
        guard(); after = final_check()
        validate_before(p, after)
        need(canonical(before) == canonical(after), 'full actual beforeafter equality before seal')
        check_sources(p['sources'])
        for e in (dict(kind='precision_after',value=after['precision']),
                  dict(kind='model_after',hashes=after['model'],classes=after['classes']),
                  dict(kind='source_after',sha256=core.digest(p['sources']))): online.event(e)
        return commit_host(online,p,rows,before,after,census,prepared,prepare_sha,guard,fixture=fixture)
    except BaseException as exc:
        # Never clear, truncate, retry or convert incomplete output to success.
        cause = online.first_transport_error
        if cause is not None:
            detail = RuntimeError('first transport error '+type(cause).__name__+': '+str(cause)+
                                  '; terminal '+type(exc).__name__+': '+str(exc))
            t.preserve_failure(rows, detail)
        else: t.preserve_failure(rows, exc)
        raise
    finally: t.close()


def run(prepared, expected, output, *, authorization):
    need(authorization == 'MAIN_APPROVED_HOST_CUDA', 'explicit MAIN-only source audit + serial GPU approval')
    start = time.monotonic(); p = verify(prepared, expected)
    out = Path(output).absolute()
    need(out.parent == ROOT/'results' and out.name.startswith(NAME) and out.resolve() == out and not out.exists(), 'exclusive own HOST directory')
    need(shutil.disk_usage(out.parent).free >= durable.TOTAL+durable.FLOOR, '4GiB plus floor before model')
    try:
        return run_loaded(p,prepared,expected,out,start)
    except BaseException as exc:
        if not out.exists():
            # Loading/static-seal failure before OnlineStore existed: retain
            # literal full denominator, no forged native call or success seal.
            out.mkdir(); durable.fsync_dir(out.parent)
            value = dict(schema=NAME+'.before_dispatch_failure',prepared=str(Path(prepared).absolute()),
                prepare_sha256=expected,rows=[dict(**x,status='blocked_before_native_dispatch',result=None)
                    for x in k2.roster()],error=dict(type=type(exc).__name__,message=str(exc)),**FLAGS)
            raw = canonical(value); need(len(raw) <= durable.RESERVE, 'before-dispatch failure reserve')
            need(shutil.disk_usage(out).free >= durable.FLOOR+len(raw), 'before-dispatch failure floor')
            with (out/'failure.json').open('xb') as f:
                need(f.write(raw) == len(raw), 'complete before-dispatch failure'); f.flush(); os.fsync(f.fileno())
            durable.fsync_dir(out)
        raise


def run_loaded(p, prepared, expected, out, start):
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    host = json_literal(require_cuda_allocation('cuda:0')); device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    guard = Guard(start, lambda: (torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device)))
    guard(); model = load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
    forwards = [(m,'forward' in m.__dict__,m.__dict__.get('forward')) for m in model.modules()]
    def snapshot():
        guard(); torch.cuda.synchronize(device)
        need(all(not m.training for m in model.modules()) and all(not v.requires_grad for v in model.parameters()), 'eval and no weight training')
        need(all(('forward' in m.__dict__) == had and m.__dict__.get('forward') is v for m,had,v in forwards), 'all native forward hooks restored')
        return dict(model=signal.model_hash(model),classes=old.model_sources(model,p),
                    precision=old.checked_precision(torch),host=host)
    with torch.autocast(device_type='cuda',enabled=False), torch.autocast(device_type='cpu',enabled=False), sdpa_kernel(SDPBackend.MATH):
        before = snapshot()
        guard.extra_check = lambda: old.checked_precision(torch)
        def factory(t,s): guard(); return old.NativeTree(model,device,t,s)
        # guard passed unchanged so it retains byte/time/peak accounting;
        # precision is checked for each native dispatch by factory and actual
        # NativeTree, plus explicit before/after MATH context snapshots.
        def final_check():
            value = snapshot(); verify(prepared,expected); guard(); return value
        return drive(p,prepared,expected,out,before,factory,final_check,guard)


class Reader(durable.Reader):
    """Own HOST manifest with explicit completion pin; frozen raw validators."""
    def __init__(self, directory, expected, completion_sha, *, allow_fixture=False):
        durable.key(expected); durable.key(completion_sha)
        directory = Path(directory).absolute()
        need(directory.resolve() == directory and not directory.is_symlink(), 'exact own HOST directory')
        need({p.name for p in directory.iterdir()} == {'cas','journal.bin','static.json','manifest.json','report.json','completion.json'}, 'whole HOST file census')
        need(codec.file_sha(durable.regular(directory/'manifest.json',durable.MANIFEST_CAP)) == expected, 'external manifest SHA')
        m = durable.read_json(directory/'manifest.json',durable.MANIFEST_CAP)
        codec.exact(m,('schema','sources','source_manifest_sha256','metadata','cas','journal','roots','capsules','report',*FLAGS))
        need(m['schema'] == NAME+'.transport' or allow_fixture and m['schema'] == NAME+'.ram_transport', 'no fixture relabel or schema spoof')
        self.fixture = m['schema'] == NAME+'.ram_transport'
        need(all(m[k] is False for k in FLAGS), 'science and production flags not signed')
        check_sources(m['sources'])
        static_path = durable.regular(directory/'static.json',durable.METADATA_CAP)
        codec.exact(m['metadata'],('bytes','sha256'))
        need(m['metadata'] == dict(bytes=static_path.stat().st_size,sha256=codec.file_sha(static_path)), 'whole static file')
        s = durable.read_json(static_path,durable.METADATA_CAP); codec.exact(s,('fields','static'))
        self.store = codec.Codec(s['fields'],s['static'])
        need(s['fields'] == self.store.fields, 'sorted unique sealed fields')
        saved_sources = s['static']['extra']['prepared']['sources']
        check_sources(saved_sources)
        need(m['source_manifest_sha256'] == core.digest(saved_sources), 'complete real source closure in sealed static')
        self.disk, self.journal = None, None
        try:
            self.disk = routing.DiskCAS.open_saved(directory/'cas',m['cas'])
            self.store.objects = self.disk; self.store.nodes_bytes = self.disk.offset+48*len(self.disk)
            j = m['journal']; codec.exact(j,('events','bytes','sha256','tail'))
            codec.uint(j['events']); need(j['events'] <= durable.EVENTS and j['bytes'] == j['events']*104, 'fixed complete journal count')
            path = durable.regular(directory/'journal.bin',durable.EVENTS*104)
            need(path.stat().st_size == j['bytes'] and codec.file_sha(path) == j['sha256'], 'whole journal SHA/bytes')
            self.journal = durable.Journal(path,readonly=True); self.store.journal = self.journal
            previous = codec.ZERO; self.native_indices = {}
            for i,raw in enumerate(self.journal):
                index, prior, ref, tail = codec.JOURNAL.unpack(raw)
                need(index == i and prior == previous and tail == codec.h(codec.CHAIN+raw[:72]) and ref in self.disk, 'complete original hashchain')
                c = codec.Cursor(self.disk[ref]); need(c.take(1) == b'E', 'event root')
                c.number('<B'); flags = c.number('<B'); slot = c.number('<B'); cid = c.option()
                need(flags in (0,1,2,3) and (slot < 6 if flags & 1 else slot == 255) and (flags & 2 or cid is None), 'literal K2 wrappers')
                if cid is not None:
                    need(flags == 3, 'explicit native slot/call'); self.native_indices.setdefault((slot,cid),[]).append(i)
                self.store.events.append(ref); self.store.membership.append({'journal'}); previous = tail
            need(previous == durable.key(j['tail']), 'final tail')
            self.store.tail = previous
            need(type(m['roots']) is dict and len(m['roots']) <= durable.ROOTS and all(len(canonical({n:v})) <= 256 for n,v in m['roots'].items()), 'finite roots')
            self.store.roots = {n:durable.key(v) for n,v in m['roots'].items()}
            need(type(m['capsules']) is dict and len(m['capsules']) <= durable.CALLS and all(len(canonical({n:v})) <= 1024 for n,v in m['capsules'].items()), 'finite capsules')
            self.store.capsules = m['capsules']
            for k in self.disk:
                raw = self.disk[k]
                if raw[:1] != b'W': continue
                c = codec.Cursor(raw); c.take(1); family = c.number('<B'); count = c.number('<I')
                need(family < 4 and 0 < count <= len(self.journal) and len(raw) == 6+8*count, 'finite real W')
                indices = [c.number('<Q') for _ in range(count)]; c.end()
                need(indices == sorted(set(indices)) and all(i < len(self.journal) for i in indices), 'ordered actual W indices')
                name = ('slot','native','capsule','ledger')[family]
                need(all(name not in self.store.membership[i] for i in indices), 'disjoint same-family views')
                self.store.views[k] = name,indices
                for i in indices: self.store.membership[i].add(name)
            report_path = durable.regular(directory/'report.json',REPORT_CAP)
            need(m['report'] == dict(bytes=report_path.stat().st_size,sha256=codec.file_sha(report_path)), 'whole report bytes')
            self.report = durable.read_json(report_path,REPORT_CAP)
            cp = durable.regular(directory/'completion.json',durable.COMPLETION_CAP)
            need(codec.file_sha(cp) == completion_sha, 'external completion SHA, including clock witness')
            w = durable.read_json(cp,durable.COMPLETION_CAP)
            codec.exact(w,('schema','manifest_sha256','report_sha256','journal_tail','wall_seconds_through_manifest_fsync',
                'boundary','excludes','CUDA_peak_bytes','artifacts_bytes_before_completion',*FLAGS))
            need(w['schema'] == NAME+'.completion' and w['manifest_sha256'] == expected and w['report_sha256'] == m['report']['sha256']
                and w['journal_tail'] == j['tail'] and all(w[k] is False for k in FLAGS), 'completion exact authority')
            need(w['boundary'] == 'manifest_report_files_and_directory_fsync_returned' and
                w['excludes'] == 'completion.json own write/fsync and process shutdown', 'explicit post-persistence wall boundary')
            wall = w['wall_seconds_through_manifest_fsync']
            need(type(wall) in (int,float) and 0 <= wall <= 7200, 'whole saved7200 boundary')
            size = sum(p.stat().st_size for p in directory.iterdir() if p.is_file())+self.disk.offset+48*len(self.disk)
            codec.uint(w['artifacts_bytes_before_completion'])
            need(w['artifacts_bytes_before_completion'] == size-cp.stat().st_size and
                 size+PREPARE_CAP+RECEIPT_CAP <= durable.TOTAL-durable.RESERVE, 'all files plus external allocations census')
            self.bytes, self.manifest, self.completion = size,m,w
            self.verify_reachability()
        except BaseException:
            self.close(); raise

    def execution(self):
        value = self.store.get(self.store.roots['host_execution'])
        codec.exact(value,('prepared','rows')); return value['prepared'], value['rows']


def validate_saved(reader, p, prepared, prepare_sha, *, allow_fixture=False):
    """Full original saved reconstruction. Never invokes an inference model."""
    r = reader.report
    codec.exact(r,('schema','status','execution_backend','prepared','prepare_sha256','rows','before','after','static_census',
        'sources_sha256','CUDA_peak_bytes','errors','wall_seconds_before_report','journal_tail','neural_executed',*FLAGS))
    fixture = reader.fixture
    need(not fixture or allow_fixture, 'explicit RAM saved opt-in')
    need(r['schema'] == NAME+('.ram_report' if fixture else '.report') and r['execution_backend'] == ('RAM_FIXTURE' if fixture else 'HOST_CUDA')
        and r['neural_executed'] is (not fixture), 'actual own backend flags')
    need(r['prepared'] == str(Path(prepared).absolute()) and r['prepare_sha256'] == prepare_sha and not r['errors']
        and r['status'] == 'six_slots_settled_not_admitted' and all(r[k] is False for k in FLAGS), 'complete report identity/status')
    saved_p, rows = reader.execution()
    need(canonical(saved_p) == canonical(p) and canonical(rows) == canonical(r['rows']), 'exact archived authority/all12')
    need(reader.manifest['source_manifest_sha256'] == r['sources_sha256'] == core.digest(p['sources']), 'entire source manifest')
    validate_before(p,r['before']); need(canonical(r['before']) == canonical(r['after']), 'frozen model/class/precision closure')
    # Recreate static seal FROM authenticated saved metadata, not from events.
    seal, _, _, _ = make_seal(p,r['before'],fixture=fixture)
    need(canonical(reader.store.static) == canonical(seal['value']) and reader.store.fields == seal['fields'], 'actual entire static authority')
    # static.json is canonical sorted JSON, while typed CAS retains original
    # dict order. Reinstall the authenticated original static root in THAT
    # order; re-encoding sorted JSON can create different dedup object counts.
    root = reader.store.get(reader.store.roots['static']); raw = canonical(root)
    need(raw == canonical(reader.store.static), 'original static root semantic bytes')
    exact_seal = dict(value=root,fields=reader.store.fields,canonical_sha256=codec.h(raw).hex(),canonical_bytes=len(raw))
    _, _, census = routing.install_static(exact_seal)
    need(r['static_census'] == census_summary(census), 'exact original typed static pack/index census')
    w = reader.completion
    need(type(r['wall_seconds_before_report']) in (int,float) and 0 <= r['wall_seconds_before_report'] <= w['wall_seconds_through_manifest_fsync'] <= 7200, 'report then post-manifest wall')
    for peak in (r['CUDA_peak_bytes'],w['CUDA_peak_bytes']):
        need(type(peak) is list and len(peak) == 2 and all(type(x) is int and 0 <= x <= cap for x,cap in zip(peak,(3*1024**3,4*1024**3))), 'saved native memory peaks')
    need(all(a <= b for a,b in zip(r['CUDA_peak_bytes'],w['CUDA_peak_bytes'])) and r['journal_tail'] == reader.manifest['journal']['tail'], 'peak monotonicity and tail')
    events = [reader.event(i) for i in range(len(reader.journal))]
    expected = {}
    for phase in ('before','after'):
        expected['source_'+phase] = dict(kind='source_'+phase,sha256=core.digest(p['sources']))
        expected['model_'+phase] = dict(kind='model_'+phase,hashes=r[phase]['model'],classes=r[phase]['classes'])
        expected['precision_'+phase] = dict(kind='precision_'+phase,value=old.precision_contract())
    for kind,e in expected.items(): need(canonical([x for x in events if x['kind'] == kind]) == canonical([e]), 'exact singleton closure journal '+kind)
    need([e['kind'] for e in events[:3]] == ['source_before','model_before','precision_before'] and
         [e['kind'] for e in events[-3:]] == ['precision_after','model_after','source_after'], 'closure brackets original execute')
    seen_capsules = set()
    def checked_result(slot, ref):
        result = reader.result(slot,ref)
        for record in result['native_calls']:
            call, cid = record['transcript'], record['call_id']
            if call and call[-1]['kind'] == 'native_call_return':
                pointer = reader.capsule(slot,cid,call)
                need(pointer['path'] not in seen_capsules, 'one unique actual capsule per returned native call')
                seen_capsules.add(pointer['path'])
        return result
    summaries = ktransport.validate_rows(p,rows,events,checked_result,reader.capsule,permit_fixture=fixture)
    need(seen_capsules == set(reader.store.capsules), 'all and only real completed native capsule descriptors')
    need(len(summaries) == 6 and all(x['status'] in original.SETTLED for x in summaries), 'all12 settled original saved mathematics')
    check_sources(p['sources'])
    return dict(schema=NAME+('.ram_consume' if fixture else '.consume'),status='saved6_k2_common_cap_engineering_verified',
        prepared=r['prepared'],prepare_sha256=prepare_sha,denominator=6,total_request_denominator=864,
        request_slots_per_arm=144,arms=[receipt_arm(s) for s in summaries],byte_proof=proof(),**FLAGS)


def consume(output, manifest_sha, completion_sha, prepared, prepare_sha):
    p = verify(prepared,prepare_sha)
    reader = Reader(output,manifest_sha,completion_sha)
    try:
        result = validate_saved(reader,p,prepared,prepare_sha)
        result.update(manifest_sha256=manifest_sha,report_sha256=reader.manifest['report']['sha256'],completion_sha256=completion_sha)
        return result
    finally: reader.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest='command',required=True)
    sub.add_parser('proof')
    q = sub.add_parser('prepare'); q.add_argument('--output',required=True)
    for cmd in ('verify','run','consume'):
        q = sub.add_parser(cmd); q.add_argument('--prepared',required=True); q.add_argument('--prepare-sha',required=True)
        if cmd != 'verify': q.add_argument('--output',required=True)
        if cmd == 'run': q.add_argument('--authorization',required=True)
        if cmd == 'consume':
            q.add_argument('--manifest-sha',required=True); q.add_argument('--completion-sha',required=True); q.add_argument('--receipt',required=True)
    a = ap.parse_args(argv)
    if a.command == 'proof': result = proof()
    elif a.command == 'prepare': result = prepare(a.output)
    elif a.command == 'verify': result = dict(verified=True,schema=verify(a.prepared,a.prepare_sha)['schema'])
    elif a.command == 'run': result = run(a.prepared,a.prepare_sha,a.output,authorization=a.authorization)
    else: result = write_new(a.receipt,consume(a.output,a.manifest_sha,a.completion_sha,a.prepared,a.prepare_sha),RECEIPT_CAP)
    print(canonical(result).decode())


if __name__ == '__main__': main()

