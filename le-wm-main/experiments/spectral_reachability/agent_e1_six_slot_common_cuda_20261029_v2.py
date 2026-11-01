"""Six-slot CUDA controller and separately metered replay; saved-only consumer.

The MAIN operator must review and authorize actual CUDA execution. No physical
dispatch or scientific admission is provided by this integration.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import argparse
import base64
import shutil
import time
import zlib
import numpy as np
from pathlib import Path

from experiments.spectral_reachability import agent_e1_six_slot_controller_core_20261029_v1 as core
from experiments.spectral_reachability import agent_e1_six_slot_phase_meter_20261029_v1 as phase
from experiments.spectral_reachability import agent_e1_six_slot_common_prepare_20261029_v1 as prep

NAME = 'agent_e1_six_slot_common_cuda_20261029_v2'
EVENT_LIMIT = 16 * 1024
PLAIN_EVENT_LIMIT = 1024 * 1024  # independent fixture-only uncompressed journal
EVENT_COUNT_LIMIT = 10000
EVENT_ARRAY_GROWTH = 32 * 1024
PUT_ARRAY_GROWTH = 1024 * 1024
PUT_LIMIT = 27
PUT_META_LIMIT = 4 * 1024 * 1024
PREP_SHA = 'cd96f6e28b25d3b7e3b6649c760b2f111747aec5c0ce9c8a41741318209df8a0'
OLD_PREP_SHA = '8c472485d44c40ce3d95f8043d2c58f9ec2694fef74786a840d1699d5a1a86bd'
ROOT = prep.ROOT
PREP_PATH = ROOT / 'results/agent_e1_six_slot_common_prepare_20261029_v1_prepare_MAIN01.json'
OLD_PREP_PATH = ROOT / 'results/agent_experiment1_common_controller_20261028_v1_prepared_MAIN01.json'
CAP = 1024 * 1024 * 1024
FLOOR = 200 * 1024 * 1024
RESERVE = 64 * 1024 * 1024
ARRAY_CAP = 384 * 1024 * 1024
META_CAP = 384 * 1024 * 1024
WALL = 900


def storage_bound():
    # zlib compressBound plus a conservative 64-byte overhead for every
    # separately synced event and final stream. JSON report/final reserve is
    # included separately, and arrays are uncompressed lossless f64 bytes.
    zmax = META_CAP + (META_CAP >> 12) + (META_CAP >> 14) + (META_CAP >> 25) + 13 + 64 * (EVENT_COUNT_LIMIT + 1)
    total = ARRAY_CAP + zmax + RESERVE
    require(total <= CAP, 'integer storage bound exceeds cap')
    event_array = EVENT_COUNT_LIMIT * EVENT_ARRAY_GROWTH
    put_array = PUT_LIMIT * PUT_ARRAY_GROWTH
    meta_bound = EVENT_COUNT_LIMIT * EVENT_LIMIT + PUT_LIMIT * PUT_META_LIMIT
    require(event_array + put_array <= ARRAY_CAP and meta_bound <= META_CAP, 'per-record census capacity')
    return dict(array_cap=ARRAY_CAP, metadata_uncompressed_cap=META_CAP,
                compressed_metadata_bound=zmax, reserved_final_bytes=RESERVE,
                worst_output_bytes=total, output_cap=CAP, disk_floor=FLOOR,
                event_cap=EVENT_COUNT_LIMIT, single_event_cap=EVENT_LIMIT,
                event_array_growth=EVENT_ARRAY_GROWTH, put_cap=PUT_LIMIT,
                put_array_growth=PUT_ARRAY_GROWTH, put_metadata_cap=PUT_META_LIMIT,
                array_record_sum_bound=event_array+put_array, metadata_record_sum_bound=meta_bound,
                qualification='enforced ceilings, not a proof that native events fit them')


def audit_packed_events(events):
    """Repack saved events exactly; independent of result/outcome selection."""
    archive = Archive(lambda raw: None)
    peak_line = peak_growth = 0
    by_kind = {}
    previous = '0'*64
    for i, event in enumerate(events):
        before = archive.size
        packed = archive.pack(event)
        body = dict(index=i,previous=previous,event=packed)
        ident = sha(canonical(body))
        line = canonical(dict(body,sha256=ident))
        growth = archive.size-before
        kind = event.get('event',event).get('kind','other')
        bucket = by_kind.setdefault(kind,dict(count=0,max_array_growth=0,max_metadata_bytes=0))
        bucket['count'] += 1
        bucket['max_array_growth'] = max(bucket['max_array_growth'],growth)
        bucket['max_metadata_bytes'] = max(bucket['max_metadata_bytes'],len(line))
        peak_line = max(peak_line,len(line)); peak_growth = max(peak_growth,growth)
        previous = ident
    return dict(event_count=len(events),max_metadata_bytes=peak_line,max_array_growth=peak_growth,
                archived_array_bytes=archive.size,by_kind=by_kind,
                qualification='observed legacy events only; does not bound new replay or learned slots')


def audit_legacy_io():
    relative, expected = prep.FILES['controller_report']
    report_path = ROOT/relative
    report = prep.parse(prep.bytes_at(report_path, expected))
    _, events = core.old.saved_payloads(report_path.parent,report)
    return dict(report_sha256=expected,events_sha256=report['artifact_sha256']['events.jsonl.gz'],
                arrays_sha256=report['artifact_sha256']['arrays.bin'],observed=audit_packed_events(events))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'


def parse_canonical(data):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'duplicate JSON key')
            out[key] = value
        return out
    def invalid(value):
        raise ValueError('nonfinite JSON: ' + value)
    obj = json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)
    require(canonical(obj) == data, 'noncanonical JSON')
    return obj


def read_bound(path, expected, limit):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), 'absolute regular input')
    with path.open('rb') as f:
        data = f.read(limit + 1)
    require(len(data) <= limit and sha(data) == expected, 'input size or SHA mismatch')
    return data


def load_bound_buffers(config_bytes, weights_bytes, instantiate, deserialize, remap, device):
    """Load from the exact buffers whose SHA is recorded; caller checks bound SHA first.

    Dependency injection lets RAM fixtures test the contract without importing torch.
    Production caller must supply torch.load(weights_only=True), Hydra instantiate,
    and the sealed checkpoint_compat remap, and must check the loaded tensor hash.
    """
    require(type(config_bytes) is bytes and type(weights_bytes) is bytes, 'immutable input buffers')
    config = json.loads(config_bytes)
    model = instantiate(config)
    state = deserialize(io.BytesIO(weights_bytes))
    remapped, changed = remap(state)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    require(not missing and not unexpected, 'checkpoint key mismatch')
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model, dict(config_sha256=sha(config_bytes), weights_sha256=sha(weights_bytes),
                       config_bytes=len(config_bytes), weights_bytes=len(weights_bytes),
                       remapped_keys=changed, strict_key_coverage=True)


class Journal:
    """Exclusive canonical JSONL, fsynced before the caller may dispatch."""
    def __init__(self, directory):
        self.directory = Path(directory)
        require(self.directory.is_absolute() and not self.directory.exists(), 'new absolute output directory')
        self.directory.mkdir()
        self.path = self.directory / 'events.jsonl'
        self.file = self.path.open('xb')
        self.index = 0
        self.previous = '0' * 64
        self.broken = False
        fd = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def event(self, event):
        require(not self.broken and not self.file.closed, 'journal closed or poisoned')
        require(self.index < EVENT_COUNT_LIMIT, 'event count cap')
        body = dict(index=self.index, previous=self.previous, event=event)
        digest = sha(canonical(body))
        data = canonical(dict(body, sha256=digest))
        require(len(data) <= PLAIN_EVENT_LIMIT, 'event byte cap')
        try:
            require(self.file.write(data) == len(data), 'short journal write')
            self.file.flush()
            os.fsync(self.file.fileno())
        except BaseException:
            self.broken = True
            raise
        self.index += 1
        self.previous = digest

    def close(self):
        self.file.close()


def read_journal(path):
    """Verify entire bounded canonical chain; partial records always fail closed."""
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'journal regular file')
    previous = '0' * 64
    events = []
    with path.open('rb') as f:
        while True:
            line = f.readline(PLAIN_EVENT_LIMIT + 1)
            if not line:
                break
            require(len(line) <= PLAIN_EVENT_LIMIT and line.endswith(b'\n'), 'partial/oversize event')
            require(len(events) < EVENT_COUNT_LIMIT, 'event count cap')
            item = parse_canonical(line)
            require(set(item) == {'index', 'previous', 'event', 'sha256'}, 'journal record schema')
            body = dict(index=len(events), previous=previous, event=item['event'])
            require(item['index'] == len(events) and item['previous'] == previous
                    and item['sha256'] == sha(canonical(body)), 'journal chain')
            previous = item['sha256']
            events.append(item['event'])
    return events, previous


def controller(task, root, arm, target, backend, sink, sync=lambda: None, check=lambda: None,
               guard=None):
    """Run the original controller using its own 144-packet actual-hook meter.

    A native caller must explicitly pass backend.guard.  Fixtures must also
    explicitly select a context manager; there is no silent unguarded default.
    """
    meter = phase.PhaseMeter('controller', sink, sync, check)
    dispatch = phase.beam.Dispatch(task, backend, meter, sink)
    require(callable(guard), 'explicit native or fixture guard required')
    with guard(meter):
        result = core.optimize(task, root, arm, target, dispatch, sink)
    require(dispatch.index == 72 and meter.counts() == phase.beam.budget(144)
            and meter.predictor_samples() == dict(attempted=144, completed=144),
            'actual controller 144-packet resource census')
    return result, meter.counts()


def verify_controller(task, root, arm, target, result, events):
    """Recompute all fixed 72 candidates and validate every recorded module IO."""
    cursor = phase.beam.Cursor(events)
    dispatch = phase.beam.SavedDispatch(task, cursor)
    rebuilt = core.optimize(task, root, arm, target, dispatch, cursor.expect)
    require(cursor.index == len(events) and dispatch.index == 72
            and len(dispatch.operations) == 576 and all(
                sum(o['resource'] == r and o['status'] == 'returned'
                    for o in dispatch.operations) == 144 for r in phase.RESOURCES)
            and rebuilt == result, 'complete controller saved reconstruction')
    return rebuilt


def replay_fixture(task, raw, backend, sink, sync=lambda: None, check=lambda: None, guard=None):
    """Exercise the independently metered four-packet replay; not a runner."""
    meter = phase.PhaseMeter('replay', sink, sync, check)
    require(callable(guard), 'explicit native or fixture guard required')
    with guard(meter):
        result = phase.replay(task, raw, backend, meter, sink)
    return result, meter.counts()


def put_new(directory, name, data):
    require(Path(name).name == name and type(data) is bytes, 'flat immutable file')
    path = Path(directory) / name
    with path.open('xb') as f:
        require(f.write(data) == len(data), 'short write')
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return sha(data)


def f32_diff(a, b):
    x, y = np.asarray(a, dtype='<f4'), np.asarray(b, dtype='<f4')
    require(x.shape == y.shape and x.size > 0 and np.isfinite(x).all() and np.isfinite(y).all(), 'finite same-shaped floats')
    # Monotonic encoding of IEEE-754 bit patterns, including signed zero.
    def ordered(v):
        u = v.view('<u4').astype(np.uint64)
        return np.where((u & 0x80000000) != 0, 0xffffffff - u, u + 0x80000000)
    u, v = ordered(x), ordered(y)
    return dict(shape=list(x.shape), bitwise_equal=bool(np.array_equal(x.view('<u4'), y.view('<u4'))),
                max_abs=float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))),
                max_ulp=int(np.max(np.maximum(u, v) - np.minimum(u, v))))


def compare_replays(rows, seal):
    require(len(rows) == 6 and [(r['root'], r['arm']) for r in rows] == list(core.SLOTS), 'all six fixed slots')
    require(sha(canonical(rows)) == seal, 'all six replay outputs sealed first')
    out = []
    for row in rows:
        require(len(row['whole']) == len(row['segmented']) == 2, 'two replay checkpoints')
        target, goal = core.old.vec(row['target']), core.old.vec(row['goal'])
        def cost(states):
            prefix = float(np.mean((core.old.vec(states[0]['latent_history'][-1])-target)**2))
            terminal = float(np.mean((core.old.vec(states[1]['latent_history'][-1])-goal)**2))
            return dict(prefix_target_mse=prefix, terminal_goal_mse=terminal, tracking_cost=.5*(prefix+terminal))
        out.append(dict(root=row['root'], arm=row['arm'], whole_cost=cost(row['whole']),
            segmented_cost=cost(row['segmented']), checkpoints=[
            {key: f32_diff(a[key], b[key]) for key in ('latent_history', 'action_history')}
            for a, b in zip(row['whole'], row['segmented'], strict=True)]))
    return out


class Archive:
    """Lossless float64 array archive; shared arrays are stored only once."""
    def __init__(self, append):
        self.append = append
        self.entries = {}
        self.size = 0

    def pack(self, value):
        if type(value) is list:
            try:
                a = np.asarray(value)
            except ValueError:
                a = np.asarray([], dtype=int)
            def all_float(v):
                return all(all_float(x) for x in v) if type(v) is list else type(v) is float
            if a.dtype == np.float64 and 10 <= a.size <= 4096 and a.ndim <= 4 and all_float(value):
                require(np.isfinite(a).all(), 'finite archived array')
                raw = a.astype('<f8').tobytes()
                ident = sha(raw)
                if ident not in self.entries:
                    require(self.size + len(raw) <= ARRAY_CAP, 'array byte ceiling')
                    self.append(raw)
                    self.entries[ident] = (self.size, len(raw))
                    self.size += len(raw)
                offset, size = self.entries[ident]
                return dict(__f64__=ident, shape=list(a.shape), offset=offset, nbytes=size)
            return [self.pack(x) for x in value]
        if type(value) is dict:
            require('__f64__' not in value, 'reserved array reference')
            return {k: self.pack(v) for k, v in value.items()}
        return value


class ArchiveReader:
    def __init__(self, raw):
        require(len(raw) <= ARRAY_CAP, 'array byte ceiling')
        self.raw, self.entries, self.used = raw, {}, set()

    def unpack(self, value):
        if type(value) is dict and '__f64__' in value:
            require(set(value) == {'__f64__', 'shape', 'offset', 'nbytes'}, 'array reference schema')
            shape, offset, n = value['shape'], value['offset'], value['nbytes']
            require(type(shape) is list and 1 <= len(shape) <= 4 and
                    all(type(x) is int and x > 0 for x in shape), 'array shape')
            size = int(np.prod(shape, dtype=np.int64))
            require(10 <= size <= 4096 and type(offset) is int and type(n) is int and
                    offset >= 0 and n == size * 8 and offset + n <= len(self.raw), 'array extent')
            data = self.raw[offset:offset+n]
            ident = value['__f64__']
            require(sha(data) == ident and (ident not in self.entries or self.entries[ident] == (offset,n)),
                    'array SHA and unique offset')
            self.entries[ident] = (offset,n)
            self.used.add((offset,n))
            a = np.frombuffer(data, dtype='<f8').reshape(shape)
            require(np.isfinite(a).all(), 'finite archived value')
            return a.tolist()
        if type(value) is dict:
            return {k:self.unpack(v) for k,v in value.items()}
        if type(value) is list:
            return [self.unpack(v) for v in value]
        return value

    def finish(self):
        end = 0
        for offset,n in sorted(self.used):
            require(offset == end, 'unused or overlapping archived data')
            end += n
        require(end == len(self.raw), 'all array bytes referenced')


class Store:
    def __init__(self, directory, started=None):
        storage_bound()
        self.path = Path(directory)
        require(self.path.is_absolute() and not self.path.exists() and
                shutil.disk_usage(self.path.parent).free >= FLOOR + CAP, 'new output and disk reservation')
        self.path.mkdir()
        self.started = time.monotonic() if started is None else started
        self.meta = self.index = self.puts = 0
        self.previous = '0' * 64
        self.closed = self.broken = False
        self.comp = zlib.compressobj(6,zlib.DEFLATED,31)
        self.write('arrays.bin', b'')
        self.write('events.jsonl.gz', b'')
        self.archive = Archive(lambda data: self.write('arrays.bin', data, append=True))

    def check(self):
        require(time.monotonic()-self.started <= WALL, 'wall limit')
        require(shutil.disk_usage(self.path).free >= FLOOR, 'disk floor')

    def write(self, name, data, append=False, final=False):
        require(Path(name).name == name and not (self.path/name).is_symlink(), 'flat output file')
        if not final:
            self.check()
        used = sum(x.stat().st_size for x in self.path.iterdir())
        require(used + len(data) <= CAP - (0 if final else RESERVE)
                and shutil.disk_usage(self.path).free >= FLOOR + len(data), 'output cap/floor')
        with (self.path/name).open('ab' if append else 'xb') as f:
            require(f.write(data) == len(data), 'short write')
            f.flush(); os.fsync(f.fileno())
        fd = os.open(self.path, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)

    def event(self, event):
        require(not self.closed and not self.broken and self.index < EVENT_COUNT_LIMIT, 'journal state/count')
        try:
            before = self.archive.size
            packed = self.archive.pack(event)
            require(self.archive.size-before <= EVENT_ARRAY_GROWTH, 'event array growth ceiling')
            body = dict(index=self.index, previous=self.previous, event=packed)
            ident = sha(canonical(body))
            raw = canonical(dict(body, sha256=ident))
            require(len(raw) <= EVENT_LIMIT and self.meta+len(raw) <= META_CAP, 'metadata ceiling')
            self.write('events.jsonl.gz', self.comp.compress(raw)+self.comp.flush(zlib.Z_SYNC_FLUSH), append=True)
            self.index += 1; self.previous = ident; self.meta += len(raw)
        except BaseException:
            self.broken = True
            raise

    def put(self, name, value):
        require(self.puts < PUT_LIMIT, 'payload count ceiling')
        before = self.archive.size
        raw = canonical(self.archive.pack(value))
        require(self.archive.size-before <= PUT_ARRAY_GROWTH and len(raw) <= PUT_META_LIMIT
                and self.meta+len(raw) <= META_CAP, 'payload growth ceiling')
        self.write(name, zlib.compress(raw))
        self.meta += len(raw); self.puts += 1

    def close(self):
        if not self.closed:
            self.closed = True
            self.write('events.jsonl.gz', self.comp.flush(zlib.Z_FINISH), append=True, final=True)


def bounded_decompress(data, limit, wbits=zlib.MAX_WBITS):
    d = zlib.decompressobj(wbits)
    raw = d.decompress(data, limit+1)
    require(len(raw) <= limit and d.eof and not d.unconsumed_tail and not d.unused_data, 'bounded complete compressed stream')
    return raw


def read_store(output, report):
    output = Path(output)
    names = set(report['artifacts'])
    require({p.name for p in output.iterdir()} == names | {'report.json','completion.json','timing.json'}, 'all output artifacts')
    for name, digest in report['artifacts'].items():
        require(Path(name).name == name and not (output/name).is_symlink() and
                (output/name).is_file() and sha((output/name).read_bytes()) == digest, 'artifact SHA')
    reader = ArchiveReader((output/'arrays.bin').read_bytes())
    events_raw = bounded_decompress((output/'events.jsonl.gz').read_bytes(), META_CAP, 31)
    require(events_raw.endswith(b'\n'), 'complete event stream')
    previous = '0' * 64
    events = []
    for line in events_raw.splitlines(keepends=True):
        require(len(line) <= EVENT_LIMIT and len(events) < EVENT_COUNT_LIMIT, 'saved event census')
        item = parse_canonical(line)
        require(set(item) == {'index','previous','event','sha256'}, 'event schema')
        body = dict(index=len(events), previous=previous, event=item['event'])
        require(item['index'] == len(events) and item['previous'] == previous and
                item['sha256'] == sha(canonical(body)), 'event chain')
        previous = item['sha256']
        events.append(reader.unpack(item['event']))
    require(len(events) == report['journal_events'] and previous == report['journal_sha256'], 'journal terminal')
    payload = {}
    meta = len(events_raw)
    payload_names = names - {'arrays.bin','events.jsonl.gz'}
    require(len(payload_names) <= PUT_LIMIT, 'saved payload census')
    for name in sorted(payload_names):
        raw = bounded_decompress((output/name).read_bytes(), META_CAP)
        meta += len(raw)
        require(len(raw) <= PUT_META_LIMIT and meta <= META_CAP, 'whole metadata census')
        payload[name] = reader.unpack(parse_canonical(raw))
    reader.finish()
    require(meta == report['metadata_bytes'] and len(reader.raw) == report['array_bytes'], 'complete byte census')
    return payload, events


def authorities(path=PREP_PATH, expected=PREP_SHA):
    require(expected == PREP_SHA and Path(path) == PREP_PATH, 'fixed six-slot prepare authority')
    six = prep.parse(prep.bytes_at(Path(path), expected))
    require(six == prep.build(), 'complete saved six-slot prepare')
    previous = prep.parse(prep.bytes_at(OLD_PREP_PATH, OLD_PREP_SHA))
    for name, expected_hash in previous['bindings_sha256'].items():
        p = Path(name)
        require(p.is_file() and not p.is_symlink() and sha(p.read_bytes()) == expected_hash, 'source closure: '+name)
    own = {
        str(ROOT/'experiments/spectral_reachability/agent_e1_six_slot_controller_core_20261029_v1.py'):
            '9b69ce4b132c99aa220fa7ea8238af35c210ee42706995d3eefccad6c340066d',
        str(ROOT/'experiments/spectral_reachability/agent_e1_six_slot_phase_meter_20261029_v1.py'):
            'deb200df9846ffed4a80f2b5a056c601005cfca83dbfc177bf67847bfdc9383f',
    }
    for name, ident in own.items():
        require(sha(Path(name).read_bytes()) == ident, 'six-slot source drift')
    return six, previous


def target_vector(row):
    raw = base64.b64decode(row['target_f32_le_base64'], validate=True)
    require(len(raw) == 768 and sha(raw) == row['target_data_sha256'], 'target raw bytes')
    a = np.frombuffer(raw, dtype='<f4')
    require(np.isfinite(a).all(), 'finite target')
    return a.astype(float).tolist()


def fixed_rows(six):
    require([(x['root_slot'],x['arm']) for x in six['rows']] == list(core.SLOTS), 'fixed six slots')
    for row in six['rows']:
        target_vector(row)
    require(sha(canonical(six['rows'])) == six['target_seal_sha256'], 'target seal')
    return six['rows']


def reload_packed(store, name):
    raw = bounded_decompress((store.path/name).read_bytes(), META_CAP)
    reader = ArchiveReader((store.path/'arrays.bin').read_bytes())
    return reader.unpack(parse_canonical(raw))


def execute_groups(six, previous, store, backend_for, guard_for, sync, check, rows):
    """All six fixed slots; no scoring of replay until all six are sealed."""
    replay_rows = []
    for i, target_row in enumerate(fixed_rows(six)):
        root, arm = core.SLOTS[i]
        slot = previous['slots'][root]
        require(slot['task_id'] == target_row['task_id'] and slot['task_sha256'] == target_row['task_sha256']
                and slot['blind_sha256'] == target_row['blind_sha256'], 'task/target binding')
        task = core.old.sampling.old.load_export(slot['task_export'], slot['task_sha256'])['gn_task']
        require(core.old.core.task_id(task) == slot['task_id'], 'loaded task identity')
        backend = backend_for(task)
        row = rows[i]
        row['status'] = 'attempted'
        def sink_controller(e): store.event(dict(slot=i, phase='controller', event=e))
        meter = phase.PhaseMeter('controller', sink_controller, sync, check)
        dispatch = phase.beam.Dispatch(task, backend, meter, sink_controller)
        try:
            with guard_for(backend, meter):
                result = core.optimize(task, root, arm, target_vector(target_row), dispatch, sink_controller)
            require(dispatch.index == 72 and meter.counts() == phase.beam.budget(144)
                    and meter.predictor_samples() == dict(attempted=144, completed=144), 'controller complete')
        finally:
            row['controller_resources'] = meter.counts()
            row['controller_samples'] = meter.predictor_samples()
        result_name = f'slot_{i}.result.z'
        store.put(result_name, result)
        plan = core.old.model_plan(slot, arm, result)
        store.put(f'slot_{i}.plan.z', plan)
        raw_name = f'slot_{i}.raw.z'
        store.put(raw_name, plan['commands'])
        raw = reload_packed(store, raw_name)
        require(raw == plan['commands'], 'literal raw10 reloaded')
        store.event(dict(kind='raw10_sealed_and_reloaded', slot=i, raw_sha256=sha(canonical(raw))))
        def sink_replay(e): store.event(dict(slot=i, phase='replay', event=e))
        replay_meter = phase.PhaseMeter('replay', sink_replay, sync, check)
        try:
            with guard_for(backend, replay_meter):
                replay = phase.replay(task, raw, backend, replay_meter, sink_replay)
        finally:
            row['replay_resources'] = replay_meter.counts()
            row['replay_samples'] = replay_meter.predictor_samples()
        require(replay_meter.counts() == phase.beam.budget(4) and
                replay_meter.predictor_samples() == dict(attempted=4, completed=4), 'paid replay complete')
        replay_name = f'slot_{i}.replay.z'
        store.put(replay_name, replay)
        store.event(dict(kind='slot_replay_sealed', slot=i, replay_sha256=sha(canonical(replay))))
        replay_rows.append(dict(root=root, arm=arm, target=target_vector(target_row), goal=task['goal'],
                                whole=replay['whole'], segmented=replay['segmented']))
        row.update(status='sealed', selection_sha256=plan['selection_sha256'])
    seal = sha(canonical(replay_rows))
    store.put('all_replays.z', replay_rows)
    store.event(dict(kind='all_replays_sealed_before_score', sha256=seal))
    scores = compare_replays(replay_rows, seal)
    store.put('replay_scores.z', scores)
    store.event(dict(kind='replay_scores_saved', sha256=sha(canonical(scores))))
    return scores


def manifest():
    six, previous = authorities()
    fixed_rows(six)
    files = [Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')]
    return dict(schema=NAME+'.prepare', six_prepare=str(PREP_PATH), six_sha256=PREP_SHA,
                old_prepare=str(OLD_PREP_PATH), old_sha256=OLD_PREP_SHA,
                sources={str(p):sha(p.read_bytes()) for p in files}, targets_sha256=six['target_seal_sha256'],
                expected_model_hash=previous['expected_model_hash'], storage=storage_bound(),
                controller_per_slot=144, replay_extra_per_slot=4, slots=6, total_per_axis=888,
                new_encoder=0, new_ad=0, new_physics=0, training=False,
                normalizer='P2 engineering, not historical Hi scaler', scientific_admission=False)


def verify_manifest(path, expected):
    path = Path(path)
    result = parse_canonical(read_bound(path, expected, 2*1024*1024))
    require(result == manifest(), 'prepare/source/authority drift')
    return result


def publish_completion(store, report_sha, clock=time.monotonic):
    """Record a post-fsync witness; witness durability cannot attest itself."""
    completion = dict(schema=NAME+'.completion', report_sha256=report_sha,
                      timing_boundary='completion file and directory fsync measured by separate timing witness')
    store.write('completion.json',canonical(completion),final=True)
    elapsed = clock()-store.started
    timing = dict(schema=NAME+'.timing',report_sha256=report_sha,
                  completion_sha256=sha((store.path/'completion.json').read_bytes()),
                  elapsed_after_completion_fsync=elapsed,wall_limit=WALL,wall_within_limit=elapsed<=WALL,
                  timing_boundary='run entry through completion file and directory fsync; excludes timing witness fsync')
    store.write('timing.json',canonical(timing),final=True)
    return completion, timing


def run(prepared, expected, output):
    started = time.monotonic()
    binding = verify_manifest(prepared, expected)
    six, previous = authorities()
    store = Store(Path(output), started=started)
    rows = [dict(root=r, arm=a, status='not_started', controller_resources=None,
                 controller_samples=None, replay_resources=None, replay_samples=None)
            for r,a in core.SLOTS]
    errors = []; before = after = host = loaded = gpu = None; model = None
    try:
        store.put('targets.z', fixed_rows(six))
        store.event(dict(kind='all_targets_sealed_before_model', sha256=six['target_seal_sha256']))
        os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from hydra.utils import instantiate
        from experiments.checkpoint_compat import remap_legacy_vit_keys
        from experiments.gpu_preflight import require_cuda_allocation
        host = require_cuda_allocation('cuda:0')
        device = torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
        policy = Path(previous['policy'])
        paths = [policy/'config.json', policy/'weights.pt']
        config = read_bound(paths[0], previous['bindings_sha256'][str(paths[0])], 4*1024*1024)
        weights = read_bound(paths[1], previous['bindings_sha256'][str(paths[1])], 512*1024*1024)
        model, loaded = load_bound_buffers(config, weights, instantiate,
            lambda stream: torch.load(stream, map_location='cpu', weights_only=True), remap_legacy_vit_keys, device)
        del weights, config
        core.old.enc.frozen(model)
        require(all(p.device.type == 'cuda' for p in model.parameters()), 'CUDA only')
        before = core.old.enc.tensor_hash(model)
        require(before == binding['expected_model_hash'], 'loaded frozen model hash')
        store.event(dict(kind='model_loaded', loaded=loaded, model_hash=before,
                         cuda=True, tf32=False, autocast=False, math_requested=True))
        def check():
            store.check()
            require(not torch.is_autocast_enabled('cuda') and not torch.backends.cuda.matmul.allow_tf32
                    and not torch.backends.cudnn.allow_tf32, 'precision guard')
            require(torch.cuda.max_memory_allocated(device) <= 4*1024**3 and
                    torch.cuda.max_memory_reserved(device) <= 4*1024**3, 'GPU memory ceiling')
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            execute_groups(six, previous, store, lambda t: phase.beam.NativeEdge(model,device,t),
                           lambda backend,meter: backend.guard(meter), lambda:torch.cuda.synchronize(device), check, rows)
        check()
    except BaseException as exc:
        errors.append(dict(type=type(exc).__name__, message=str(exc)[:2048]))
        for row in rows:
            if row['status'] == 'attempted': row['status'] = 'failed_partial'
        try: store.event(dict(kind='execution_error', error=errors[0]))
        except BaseException as secondary: errors.append(dict(type=type(secondary).__name__, message='journal: '+str(secondary)[:2048]))
    finally:
        if model is not None:
            try:
                torch.cuda.synchronize(device)
                gpu = dict(peak_allocated=int(torch.cuda.max_memory_allocated(device)),
                           peak_reserved=int(torch.cuda.max_memory_reserved(device)), cap=4*1024**3,
                           requested_math=True, observed_math=None,
                           tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
                           tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
                           autocast=bool(torch.is_autocast_enabled('cuda')))
                core.old.enc.frozen(model); after = core.old.enc.tensor_hash(model)
            except BaseException as exc: errors.append(dict(type=type(exc).__name__,message='final model: '+str(exc)[:2048]))
        try: verify_manifest(prepared, expected)
        except BaseException as exc: errors.append(dict(type=type(exc).__name__,message='final source: '+str(exc)[:2048]))
        try: store.close()
        except BaseException as exc: errors.append(dict(type=type(exc).__name__,message='journal close: '+str(exc)[:2048]))
        hashes = {p.name:sha(p.read_bytes()) for p in store.path.iterdir() if p.is_file() and not p.is_symlink()}
        complete = (not errors and before == after == binding['expected_model_hash'] and
                    all(r['status']=='sealed' for r in rows))
        report = dict(schema=NAME+'.report', status='all_six_sealed' if complete else 'partial_or_failed',
            prepared=str(Path(prepared)), prepare_sha256=expected, rows=rows, errors=errors,
            artifacts=hashes, journal_events=store.index, journal_sha256=store.previous,
            array_bytes=store.archive.size, metadata_bytes=store.meta, model_hash_before=before,
            model_hash_after=after, loaded=loaded, host=host, gpu=gpu, storage=storage_bound(),
            elapsed_before_report=time.monotonic()-store.started, scientific_admission=False,
            training=False, new_physics=0, new_encoder=0, new_ad=0)
        store.write('report.json',canonical(report),final=True)
        completion, timing = publish_completion(store, sha((store.path/'report.json').read_bytes()))
    require(complete and timing['wall_within_limit'], 'partial/failed evidence retained; no retry')
    return dict(report=str(store.path/'report.json'), report_sha256=completion['report_sha256'],
                completion=str(store.path/'completion.json'), completion_sha256=timing['completion_sha256'],
                timing=str(store.path/'timing.json'), timing_sha256=sha(canonical(timing)))


def consume(output, report_sha, completion_sha, timing_sha):
    output = Path(output)
    require(output.is_absolute() and output.is_dir() and not output.is_symlink(), 'output directory')
    report = parse_canonical(read_bound(output/'report.json', report_sha, RESERVE))
    completion = parse_canonical(read_bound(output/'completion.json', completion_sha, RESERVE))
    timing = parse_canonical(read_bound(output/'timing.json', timing_sha, RESERVE))
    binding = verify_manifest(report['prepared'], report['prepare_sha256'])
    six, previous = authorities()
    require(report['schema'] == NAME+'.report' and report['status'] == 'all_six_sealed'
            and report['errors'] == [] and report['scientific_admission'] is False
            and report['training'] is False and report['new_physics'] == report['new_encoder'] == report['new_ad'] == 0
            and report['storage'] == storage_bound() and report['model_hash_before'] == report['model_hash_after'] == binding['expected_model_hash']
            and completion == dict(schema=NAME+'.completion', report_sha256=report_sha,
                timing_boundary='completion file and directory fsync measured by separate timing witness')
            and timing == dict(schema=NAME+'.timing', report_sha256=report_sha, completion_sha256=completion_sha,
                elapsed_after_completion_fsync=timing['elapsed_after_completion_fsync'], wall_limit=WALL,
                wall_within_limit=True, timing_boundary='run entry through completion file and directory fsync; excludes timing witness fsync')
            and type(timing['elapsed_after_completion_fsync']) in (int,float)
            and type(report['elapsed_before_report']) in (int,float)
            and 0 <= report['elapsed_before_report'] <= timing['elapsed_after_completion_fsync'] <= WALL,
            'complete report/completion/timing')
    require(sum(p.stat().st_size for p in output.iterdir()) <= CAP, 'total actual byte cap')
    gpu = report['gpu']
    require(gpu['requested_math'] is True and gpu['observed_math'] is None and
            gpu['tf32_matmul'] is gpu['tf32_cudnn'] is gpu['autocast'] is False and
            gpu['cap'] == 4*1024**3 and 0 <= gpu['peak_allocated'] <= gpu['cap'] and
            0 <= gpu['peak_reserved'] <= gpu['cap'], 'actual GPU meter and precision boundary')
    expected_files = {'arrays.bin','events.jsonl.gz','targets.z','all_replays.z','replay_scores.z'} | {
        f'slot_{i}.{kind}.z' for i in range(6) for kind in ('result','plan','raw','replay')}
    require(set(report['artifacts']) == expected_files, 'complete artifact slots')
    payload, events = read_store(output, report)
    targets = fixed_rows(six)
    require(payload['targets.z'] == targets and events[0] == dict(kind='all_targets_sealed_before_model',sha256=six['target_seal_sha256']), 'all targets sealed before load')
    loaded = report['loaded']
    policy = Path(previous['policy'])
    require(loaded['config_sha256'] == previous['bindings_sha256'][str(policy/'config.json')]
            and loaded['weights_sha256'] == previous['bindings_sha256'][str(policy/'weights.pt')]
            and loaded['strict_key_coverage'] is True
            and events[1] == dict(kind='model_loaded',loaded=loaded,model_hash=binding['expected_model_hash'],
                                  cuda=True,tf32=False,autocast=False,math_requested=True), 'model load attestation')
    require(len(report['rows']) == 6, 'six report rows')
    at = 2; replay_rows = []
    for i,(root,arm) in enumerate(core.SLOTS):
        row = report['rows'][i]; slot = previous['slots'][root]
        require(row['root'] == root and row['arm'] == arm and row['status'] == 'sealed'
                and row['controller_resources'] == phase.beam.budget(144)
                and row['replay_resources'] == phase.beam.budget(4)
                and row['controller_samples'] == dict(attempted=144,completed=144)
                and row['replay_samples'] == dict(attempted=4,completed=4), 'complete per-slot resources')
        task = core.old.sampling.old.load_export(slot['task_export'],slot['task_sha256'])['gn_task']
        require(core.old.core.task_id(task) == targets[i]['task_id'] == slot['task_id'], 'task identity')
        group=[]
        while at < len(events) and events[at].get('slot') == i and events[at].get('phase') == 'controller':
            require(set(events[at]) == {'slot','phase','event'}, 'controller scope')
            group.append(events[at]['event']); at += 1
        result = payload[f'slot_{i}.result.z']
        verify_controller(task,root,arm,target_vector(targets[i]),result,group)
        plan = core.old.model_plan(slot,arm,result)
        raw = payload[f'slot_{i}.raw.z']
        require(plan == payload[f'slot_{i}.plan.z'] and raw == plan['commands'] and
                row['selection_sha256'] == plan['selection_sha256'] and
                events[at] == dict(kind='raw10_sealed_and_reloaded',slot=i,raw_sha256=sha(canonical(raw))), 'literal selected action')
        at += 1; group=[]
        while at < len(events) and events[at].get('slot') == i and events[at].get('phase') == 'replay':
            require(set(events[at]) == {'slot','phase','event'}, 'replay scope')
            group.append(events[at]['event']); at += 1
        replay = phase.verify_saved_replay(task,raw,group)
        require(replay == payload[f'slot_{i}.replay.z'] and events[at] ==
                dict(kind='slot_replay_sealed',slot=i,replay_sha256=sha(canonical(replay))), 'replay seal')
        at += 1
        replay_rows.append(dict(root=root,arm=arm,target=target_vector(targets[i]),goal=task['goal'],
                                whole=replay['whole'],segmented=replay['segmented']))
    seal = sha(canonical(replay_rows))
    require(payload['all_replays.z'] == replay_rows and events[at] == dict(kind='all_replays_sealed_before_score',sha256=seal), 'six replay barrier')
    scores = compare_replays(replay_rows,seal); at += 1
    require(payload['replay_scores.z'] == scores and events[at] == dict(kind='replay_scores_saved',sha256=sha(canonical(scores)))
            and at+1 == len(events), 'whole replay scores and no extra events')
    verify_manifest(report['prepared'], report['prepare_sha256'])
    return dict(schema=NAME+'.consume',status='all_six_saved_controller_and_replays_verified',
                report_sha256=report_sha, completion_sha256=completion_sha, timing_sha256=timing_sha,
                elapsed_after_completion_fsync=timing['elapsed_after_completion_fsync'], rows=report['rows'], scores=scores,
                controller_per_slot=144,replay_per_slot=4,scientific_admission=False,
                evidence_boundary='saved module IO and loader attestation; no neural forward re-execution')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify','run','consume','audit-io'))
    parser.add_argument('--output'); parser.add_argument('--prepared'); parser.add_argument('--prepare-sha')
    parser.add_argument('--report-sha'); parser.add_argument('--completion-sha'); parser.add_argument('--timing-sha')
    parser.add_argument('--receipt')
    args = parser.parse_args()
    if args.command == 'audit-io':
        result = audit_legacy_io()
    elif args.command == 'prepare':
        require(args.output is not None, 'output required')
        path = Path(args.output); require(path.is_absolute(), 'absolute path')
        data = canonical(manifest()); digest = put_new(path.parent,path.name,data)
        result = dict(path=str(path),sha256=digest)
    elif args.command == 'verify':
        verify_manifest(args.prepared,args.prepare_sha); result = dict(status='prepare_verified')
    elif args.command == 'run':
        result = run(args.prepared,args.prepare_sha,args.output)
    else:
        result = consume(args.output,args.report_sha,args.completion_sha,args.timing_sha)
        require(args.receipt is not None, 'exclusive receipt output required')
        path = Path(args.receipt); require(path.is_absolute(), 'absolute receipt path')
        result = dict(path=str(path), sha256=put_new(path.parent,path.name,canonical(result)))
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__ == '__main__':
    main()
