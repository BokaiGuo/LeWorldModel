"""Prospective bounded K17 first5 storage primitive; not a physical runner."""
import hashlib
import json
import os
from pathlib import Path
import shutil

from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as transport

MIB = 1024 ** 2
CAP = 128 * MIB
FLOOR = 200 * MIB
FINAL_RESERVE = 8 * MIB
PNG_COUNT = 306
PNG_MAX = transport.PNG_BOUND
LIMITS = {'journal.jsonl': 32 * MIB, 'execution_binding.json': 2 * MIB,
          'report.json': 16 * MIB, 'completion.json': 2 * MIB}


def census():
    png = PNG_COUNT * PNG_MAX
    ordinary = png + sum(LIMITS.values())
    return {'png_count': PNG_COUNT, 'png_max_each': PNG_MAX, 'png_max_total': png,
            'file_limits': dict(LIMITS), 'ordinary_max': ordinary,
            'failure_reserve': FINAL_RESERVE, 'total_max': ordinary + FINAL_RESERVE,
            'cap': CAP, 'floor': FLOOR,
            'fits': ordinary + FINAL_RESERVE <= CAP,
            'scope': 'enforced byte quotas, not a guarantee every native record fits'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


class Store:
    """Exclusive dir; enforce per-file, aggregate, floor and failure reserve.

    Write errors permanently fault ordinary writes. The reserved failure record
    may still be attempted; this cannot guarantee persistence after I/O failure.
    """
    def __init__(self, directory):
        self.path = Path(directory)
        if self.path.exists() or self.path.is_symlink() or not self.path.parent.is_dir():
            raise ValueError('exclusive output directory')
        if shutil.disk_usage(self.path.parent).free < CAP + FLOOR:
            raise ValueError('initial floor')
        self.path.mkdir()
        fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self.index = 0
        self.previous = None
        self.fault = False

    def _entries(self):
        entries = list(self.path.iterdir())
        if any(not p.is_file() or p.is_symlink() for p in entries):
            raise ValueError('unexpected output entry')
        return entries

    def _write(self, name, data, *, append=False, failure=False):
        if self.fault and not failure:
            raise ValueError('store fault')
        if type(data) is not bytes or Path(name).name != name:
            raise ValueError('invalid output')
        entries = self._entries()
        pngs = [p for p in entries if p.suffix == '.png']
        path = self.path / name
        if failure:
            if name != 'failure.json' or append or len(data) > FINAL_RESERVE:
                raise ValueError('failure reserve')
            limit = FINAL_RESERVE
        elif name.endswith('.png'):
            if append or len(pngs) >= PNG_COUNT or len(data) > PNG_MAX:
                raise ValueError('PNG cap')
            limit = PNG_MAX
        else:
            if name not in LIMITS or append != (name == 'journal.jsonl'):
                raise ValueError('file contract')
            limit = LIMITS[name]
        current = path.stat().st_size if path.exists() else 0
        if (not append and current) or current + len(data) > limit:
            if not failure:
                self.fault = True
            raise ValueError('file quota')
        total = sum(p.stat().st_size for p in entries)
        ceiling = CAP if failure else CAP - FINAL_RESERVE
        if total + len(data) > ceiling or shutil.disk_usage(self.path).free < FLOOR + len(data):
            if not failure:
                self.fault = True
            raise ValueError('aggregate quota/floor')
        try:
            with path.open('ab' if append else 'xb') as stream:
                if stream.write(data) != len(data):
                    raise OSError('short write')
                stream.flush()
                os.fsync(stream.fileno())
            fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except BaseException:
            self.fault = True
            raise

    def png(self, name, data):
        if not name.endswith('.png'):
            raise ValueError('PNG suffix')
        self._write(name, data)

    def put(self, name, obj):
        self._write(name, canonical(obj) + b'\n')

    def emit(self, kind, payload):
        body = {'index': self.index, 'previous': self.previous, 'kind': kind, 'payload': payload}
        event = dict(body, sha256=hashlib.sha256(canonical(body)).hexdigest())
        self._write('journal.jsonl', canonical(event) + b'\n', append=True)
        self.index += 1
        self.previous = event['sha256']
        return event

    def fail(self, record):
        self._write('failure.json', canonical(record) + b'\n', failure=True)
