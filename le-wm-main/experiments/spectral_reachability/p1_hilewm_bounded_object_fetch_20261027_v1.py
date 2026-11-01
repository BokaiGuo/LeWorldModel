"""Exclusive, bounded official ZIP-member fetch and non-executing static scan."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import pickletools
import shutil
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from dataclasses import asdict, dataclass

MIB = 1024 * 1024
DOWNLOAD_CAP = 120 * MIB
DISK_FLOOR = 200 * MIB
RANGE_CHUNK = 4 * MIB
ARCHIVE_BYTES = 2299863924
URL = 'https://zenodo.org/api/records/21353240/files/package.zip/content'
MEMBER = 'package/checkpoints/pusht/main/pusht_hi_lewm_epoch15_object.ckpt'
TAIL_SHA = '43a0c87d4f1b652caccbc26ac5e1b1ed73c0c6cf97c8e00683ecc1bdd8c16aef'
DIRECTORY_SHA = '75e0036bcb4014b30c297583fb1d96983cdcc3dbfa09a77fcf7f058e73eda67b'
PRIOR_REPORT_SHA = '7feaf480934462619149c72e293de7df09cf55835f1001a8f7d88f9194b26c66'


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(MIB), b''): h.update(block)
    return h.hexdigest()


def safe_member(name: str) -> None:
    if (not name or PurePosixPath(name).is_absolute() or '\\' in name or '\x00' in name
            or any(p in ('', '.', '..') for p in name.split('/'))):
        raise ValueError('Unsafe archive member')


@dataclass(frozen=True)
class Spec:
    name: str
    offset: int
    packed: int
    expanded: int
    crc: int
    flags: int
    method: int


def binding(project: Path) -> tuple[Spec, dict]:
    prior = project / 'results/p1_learned_subgoal_remote_inventory_20261027_v1'
    report = (prior / 'report.json').read_bytes()
    tail = (prior / 'archive_tail.bin').read_bytes()
    if digest(report) != PRIOR_REPORT_SHA or digest(tail) != TAIL_SHA:
        raise ValueError('Frozen directory provenance changed')
    end = tail.rfind(b'PK\x05\x06')
    if end < 0: raise ValueError('Missing EOCD')
    _, disk, cd_disk, on_disk, count, size, offset, comment = struct.unpack_from('<4s4H2IH', tail, end)
    if (disk or cd_disk or on_disk != count or count != 414
            or end + 22 + comment != len(tail)):
        raise ValueError('EOCD binding mismatch')
    directory = tail[offset - 2299798367:offset - 2299798367 + size]
    if digest(directory) != DIRECTORY_SHA: raise ValueError('Directory SHA mismatch')
    pos = 0
    found = []
    n = 0
    while pos < len(directory):
        f = struct.unpack_from('<4s6H3I5H2I', directory, pos)
        if f[0] != b'PK\x01\x02': raise ValueError('Central signature')
        nl, el, cl = f[10:13]
        name = directory[pos+46:pos+46+nl].decode('utf-8' if f[3] & 0x800 else 'cp437')
        if name == MEMBER:
            safe_member(name)
            if (f[15] >> 16) & 0o170000 == 0o120000: raise ValueError('Symlink target')
            found.append(Spec(name, f[16], f[8], f[9], f[7], f[3], f[4]))
        pos += 46 + nl + el + cl
        n += 1
    if n != count or pos != len(directory) or len(found) != 1:
        raise ValueError('Directory count/duplicate mismatch')
    spec = found[0]
    if (spec.offset, spec.packed, spec.expanded, spec.crc, spec.method) != (
            634194328, 113104429, 122424517, 0xd59f865b, 8):
        raise ValueError('Wrong fixed target')
    return spec, {'prior_report_sha256': digest(report), 'tail_sha256': digest(tail),
                  'directory_sha256': digest(directory)}


class Journal:
    def __init__(self, path: Path):
        self.file = path.open('x', encoding='utf-8')
        self.previous = '0' * 64
        self.sequence = 0

    def add(self, event: str, **fields) -> None:
        row = {'sequence': self.sequence, 'previous_sha256': self.previous,
               'event': event, 'time_ns': time.time_ns(), **fields}
        raw = json.dumps(row, sort_keys=True, separators=(',', ':')).encode()
        self.previous = digest(raw)
        line = json.dumps({'record': row, 'sha256': self.previous}) + '\n'
        space(Path(self.file.name).parent, len(line.encode()))
        self.file.write(line)
        self.file.flush()
        os.fsync(self.file.fileno())
        self.sequence += 1

    def close(self): self.file.close()


def space(path: Path, needed: int = 0) -> None:
    if shutil.disk_usage(path).free - needed < DISK_FLOOR:
        raise OSError('200MiB disk floor would be breached')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs): return None


class RangeReader:
    def __init__(self, journal: Journal, opener=None):
        self.journal = journal
        self.opener = opener or urllib.request.build_opener(NoRedirect())
        self.reserved = 0
        self.received = 0
        self.attempted = 0
        self.completed = 0
        self.validators = {}
        self.journal_errors = []

    def read(self, start: int, length: int, consume) -> None:
        if start < 0 or length <= 0 or start + length > ARCHIVE_BYTES:
            raise ValueError('Invalid range')
        if self.reserved + length > DOWNLOAD_CAP: raise ValueError('120MiB network cap')
        self.reserved += length
        end = start + length - 1
        self.attempted += 1
        self.journal.add('range_attempt', attempt=self.attempted, start=start, end=end)
        req = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}',
            'Accept-Encoding': 'identity', 'User-Agent': 'LeWM-bounded-object-v1'})
        try:
            response = self.opener.open(req, timeout=30)
        except urllib.error.HTTPError as exc:
            response = exc
        h = hashlib.sha256()
        actual = 0
        with response:
            headers = {k.lower(): v for k, v in response.headers.items() if k.lower() in (
                'content-range', 'content-length', 'content-encoding', 'content-type',
                'etag', 'last-modified', 'date', 'accept-ranges')}
            self.journal.add('range_headers', status=response.status, headers=headers)
            if response.geturl() != URL: raise ValueError('Redirect/host/path change forbidden')
            if response.status != 206: raise ValueError('Non-206; body not read')
            if response.headers.get('Content-Range') != f'bytes {start}-{end}/{ARCHIVE_BYTES}':
                raise ValueError('Content-Range mismatch; body not read')
            if response.headers.get('Content-Encoding', 'identity').lower() != 'identity':
                raise ValueError('Encoded response forbidden')
            if response.headers.get('Content-Length') != str(length):
                raise ValueError('Content-Length mismatch; body not read')
            for key in ('etag', 'last-modified'):
                value = headers.get(key)
                if value is not None:
                    if key in self.validators and self.validators[key] != value:
                        raise ValueError('Archive validator changed')
                    self.validators[key] = value
            try:
                while actual < length:
                    block = response.read(min(65536, length - actual))
                    if not block: raise ValueError('Truncated range')
                    actual += len(block)
                    self.received += len(block)
                    h.update(block)
                    consume(block)
                self.completed += 1
                self.journal.add('range_completed', actual=actual, body_sha256=h.hexdigest(),
                                 total_received=self.received)
            except BaseException:
                try:
                    self.journal.add('range_partial', actual=actual, body_sha256=h.hexdigest(),
                                     total_received=self.received)
                except BaseException as secondary:
                    self.journal_errors.append({'phase': 'range_partial_journal',
                        'type': type(secondary).__name__, 'message': str(secondary)})
                raise

    def small(self, start: int, length: int) -> bytes:
        if length > 131100: raise ValueError('Header/tail memory cap')
        parts = []
        self.read(start, length, parts.append)
        return b''.join(parts)


def local_header(spec: Spec, reader: RangeReader) -> int:
    safe_member(spec.name)
    header = reader.small(spec.offset, 30)
    f = struct.unpack('<4s5H3I2H', header)
    if (f[0] != b'PK\x03\x04' or f[2] != spec.flags or f[3] != spec.method
            or spec.flags & 1 or spec.method != 8):
        raise ValueError('Invalid local header flags/method')
    if not spec.flags & 8 and (f[6], f[7], f[8]) != (spec.crc, spec.packed, spec.expanded):
        raise ValueError('Local CRC/size disagreement')
    nl, el = f[9:11]
    names = reader.small(spec.offset + 30, nl + el)
    encoding = 'utf-8' if spec.flags & 0x800 else 'cp437'
    if names[:nl] != spec.name.encode(encoding): raise ValueError('Local member name mismatch')
    return spec.offset + 30 + nl + el


class InflateSink:
    def __init__(self, file, spec: Spec, output: Path, check_space=space):
        self.file, self.spec, self.output = file, spec, output
        self.check_space = check_space
        self.decoder = zlib.decompressobj(-15)
        self.packed = 0
        self.expanded = 0
        self.crc = 0
        self.raw_sha = hashlib.sha256()
        self.packed_sha = hashlib.sha256()

    def feed(self, block: bytes) -> None:
        self.packed += len(block)
        if self.packed > self.spec.packed: raise ValueError('Compressed size cap')
        self.packed_sha.update(block)
        pending = block
        while pending:
            raw = self.decoder.decompress(pending, min(MIB, self.spec.expanded-self.expanded+1))
            pending = self.decoder.unconsumed_tail
            if self.decoder.unused_data: raise ValueError('Trailing DEFLATE bytes')
            if self.expanded + len(raw) > self.spec.expanded: raise ValueError('Expansion cap')
            if raw:
                self.check_space(self.output, len(raw))
                self.file.write(raw)
                self.expanded += len(raw)
                self.crc = zlib.crc32(raw, self.crc)
                self.raw_sha.update(raw)

    def finish(self) -> dict:
        if (self.packed != self.spec.packed or self.expanded != self.spec.expanded
                or not self.decoder.eof or self.decoder.unused_data
                or self.decoder.unconsumed_tail or self.crc & 0xffffffff != self.spec.crc):
            raise ValueError('Final compressed/expanded size, DEFLATE or CRC mismatch')
        return {'compressed_bytes': self.packed, 'expanded_bytes': self.expanded,
                'crc32': f'{self.crc & 0xffffffff:08x}',
                'compressed_sha256': self.packed_sha.hexdigest(), 'object_sha256': self.raw_sha.hexdigest()}


def pickle_inventory(raw: bytes) -> dict:
    """Decode opcodes, never reconstruct objects or import GLOBAL names."""
    globals_seen = set()
    stack = []
    memo = {}
    unresolved = 0
    counts = {}
    operations = 0
    error = None
    consumed = 0
    try:
        for op, arg, pos in pickletools.genops(raw):
            operations += 1
            if operations > 1000000: raise ValueError('Opcode cap')
            name = op.name
            counts[name] = counts.get(name, 0) + 1
            consumed = pos + 1
            if name in ('UNICODE', 'BINUNICODE', 'SHORT_BINUNICODE', 'BINUNICODE8'):
                stack.append(arg)
            elif name in ('GLOBAL', 'INST'):
                globals_seen.add(str(arg))
                stack.append(None)
            elif name == 'STACK_GLOBAL':
                if len(stack) >= 2 and all(isinstance(x, str) for x in stack[-2:]):
                    globals_seen.add(stack[-2] + ' ' + stack[-1])
                else:
                    unresolved += 1
                stack = [None]
            elif name in ('BINPUT', 'LONG_BINPUT', 'PUT'):
                memo[int(arg)] = stack[-1] if stack else None
            elif name == 'MEMOIZE':
                memo[len(memo)] = stack[-1] if stack else None
            elif name in ('BINGET', 'LONG_BINGET', 'GET'):
                stack.append(memo.get(int(arg)))
            elif name not in ('PROTO', 'FRAME', 'STOP'):
                # Conservative: do not emulate reducers, persistent IDs or constructors.
                stack = [None]
            if len(globals_seen) > 10000: raise ValueError('Global count cap')
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
    return {'globals_referenced': sorted(globals_seen), 'unresolved_stack_globals': unresolved,
            'opcode_counts': counts, 'operations': operations, 'parser_error': error,
            'consumed_bytes': consumed, 'semantic_execution': False,
            'class_list_complete': False, 'note': 'References only; no reducer execution or safety certificate'}


def static_inventory(path: Path) -> dict:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > 10000: raise ValueError('Container entry count cap')
            rows, pickles = [], []
            total_pickle = 0
            names = set()
            for info in infos:
                name = info.filename.rstrip('/')
                safe_member(name)
                if name in names: raise ValueError('Duplicate inner member')
                names.add(name)
                if info.flag_bits & 1: raise ValueError('Encrypted inner member')
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Symlink inner member')
                rows.append({'path': info.filename, 'bytes': info.file_size,
                             'compressed_bytes': info.compress_size, 'crc32': f'{info.CRC:08x}'})
                if name.endswith('.pkl'):
                    total_pickle += info.file_size
                    if total_pickle > 8*MIB: raise ValueError('Static pickle byte cap')
                    with archive.open(info) as f: raw = f.read(info.file_size+1)
                    if len(raw) != info.file_size: raise ValueError('Pickle size mismatch')
                    pickles.append({'path': name, 'sha256': digest(raw), **pickle_inventory(raw)})
            return {'container': 'zip', 'entries': rows, 'pickles': pickles,
                    'tensor_storage_read': False, 'unpickled': False}
    # Legacy torch serialization: only a bounded opcode prefix; never read storages.
    with path.open('rb') as f: raw = f.read(8*MIB)
    streams = []
    offset = 0
    for _ in range(8):
        if raw[offset:offset+1] != b'\x80': break
        scan = pickle_inventory(raw[offset:])
        streams.append({'offset': offset, **scan})
        if scan['parser_error'] or not scan['consumed_bytes']: break
        offset += scan['consumed_bytes']
    return {'container': 'legacy_or_unknown', 'prefix_bytes': len(raw), 'pickles': streams,
            'unpickled': False, 'complete_inventory': False}


def fetch_object(project: Path, output: Path, opener=None) -> dict:
    spec, provenance = binding(project)
    parent = output.parent.resolve(strict=True)
    results = (project / 'results').resolve(strict=True)
    if parent != results: raise ValueError('Output must be a NEW direct child of project/results')
    space(parent, spec.expanded + 8*MIB)
    output.mkdir(exist_ok=False)
    journal = Journal(output / 'events.jsonl')
    report = {'status': 'started', 'spec': asdict(spec), 'provenance': provenance,
              'source_sha256': file_digest(Path(__file__)), 'first_error': None,
              'secondary_errors': [], 'neural_calls': 0, 'physics_calls': 0, 'unpickled': False}
    reader = RangeReader(journal, opener)
    sink = None
    original = None
    try:
        journal.add('start', spec=asdict(spec), source_sha256=report['source_sha256'])
        if digest(reader.small(2299798367, 65557)) != TAIL_SHA:
            raise ValueError('Remote archive tail changed')
        data_offset = local_header(spec, reader)
        report['data_offset'] = data_offset
        with (output/'download.part').open('xb') as f:
            sink = InflateSink(f, spec, output)
            offset = data_offset
            remaining = spec.packed
            while remaining:
                length = min(RANGE_CHUNK, remaining)
                reader.read(offset, length, sink.feed)
                f.flush()
                os.fsync(f.fileno())
                journal.add('payload_checkpoint', packed=sink.packed, expanded=sink.expanded,
                            partial_sha256=sink.raw_sha.hexdigest())
                offset += length
                remaining -= length
            verified = sink.finish()
        if digest(reader.small(2299798367, 65557)) != TAIL_SHA:
            raise ValueError('Post-download archive tail changed')
        if file_digest(output/'download.part') != verified['object_sha256']:
            raise ValueError('Persisted object SHA mismatch')
        report['verified_object'] = verified
        journal.add('object_verified', **verified)
        # Atomic exclusive publication; retain same-inode staging alias, no deletion.
        final = output / 'pusht_hi_lewm_epoch15_object.ckpt'
        os.link(output/'download.part', final)
        fsync_directory(output)
        report['artifact'] = final.name
        report['status'] = 'object_verified_static_pending'
        inventory = static_inventory(final)
        space(output, MIB)
        with (output/'static_inventory.json').open('x') as f:
            json.dump(inventory, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
        report['static_inventory_sha256'] = file_digest(output/'static_inventory.json')
        report['status'] = 'object_verified_static_recorded_not_loaded'
    except BaseException as exc:
        original = exc
        report['first_error'] = {'type': type(exc).__name__, 'message': str(exc)}
        report['status'] = 'failed_preserved'
    finally:
        report['network'] = {'attempted': reader.attempted, 'completed': reader.completed,
                             'reserved_bytes': reader.reserved, 'received_bytes': reader.received,
                             'cap_bytes': DOWNLOAD_CAP, 'validators': reader.validators}
        if sink:
            report['partial'] = {'compressed_bytes': sink.packed, 'expanded_bytes': sink.expanded,
                                  'object_prefix_sha256': sink.raw_sha.hexdigest()}
        report['secondary_errors'].extend(reader.journal_errors)
        for phase, action in (
            ('final_source_hash', lambda: report.update(source_unchanged=file_digest(Path(__file__))==report['source_sha256'])),
            ('final_journal', lambda: journal.add('terminal', status=report['status'], first_error=report['first_error'])),
            ('journal_close', journal.close),
        ):
            try: action()
            except BaseException as exc:
                report['secondary_errors'].append({'phase': phase, 'type': type(exc).__name__, 'message': str(exc)})
        if report['secondary_errors'] or report.get('source_unchanged') is not True:
            report['status'] = 'failed_preserved'
        try:
            space(output, 65536)
            with (output/'report.json').open('x') as f:
                json.dump(report, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
            fsync_directory(output)
        except BaseException as exc:
            if original is not None: raise original from exc
            raise
    if original is not None: raise original
    if report['status'] == 'failed_preserved': raise RuntimeError('Finalization failed; inspect report')
    return report


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'fetch', 'static'))
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--object', type=Path)
    parser.add_argument('--acknowledge-network-download', action='store_true')
    args = parser.parse_args()
    if args.mode == 'plan':
        spec, provenance = binding(args.project.resolve())
        print(json.dumps({'spec': asdict(spec), 'provenance': provenance,
            'download_cap': DOWNLOAD_CAP, 'disk_floor': DISK_FLOOR,
            'preflight_free_required': spec.expanded+8*MIB+DISK_FLOOR,
            'network_executed': False}, indent=2))
    elif args.mode == 'fetch':
        if not args.acknowledge_network_download or args.output is None:
            parser.error('fetch requires NEW --output and --acknowledge-network-download; host approval also required')
        print(json.dumps(fetch_object(args.project.resolve(), args.output.absolute()), indent=2))
    else:
        if args.object is None: parser.error('static requires --object')
        print(json.dumps(static_inventory(args.object), indent=2))


if __name__ == '__main__': main()
