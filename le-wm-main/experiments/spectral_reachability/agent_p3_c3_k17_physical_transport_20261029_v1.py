"""Bounded K17 transport seam. Not a native runner or a dispatch authority."""
import hashlib
import json
import os
from pathlib import Path

CAP = 64 * 1024**2
RESERVE = 8 * 1024**2
PNG_COUNT = 306
SCAN = 224 * (224 * 3 + 1)
PNG_BOUND = SCAN + (SCAN >> 12) + (SCAN >> 14) + (SCAN >> 25) + 13 + 1024


def census():
    png = PNG_COUNT * PNG_BOUND
    return {'png_count': PNG_COUNT, 'png_bound_each': PNG_BOUND,
            'png_bound_total': png, 'cap': CAP, 'reserve': RESERVE,
            'metadata_remaining': CAP - RESERVE - png,
            'metadata_sufficiency_proved': False}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_journal(path):
    """Read strict canonical, newline-terminated, hash-chained events."""
    raw = Path(path).read_bytes()
    if raw and not raw.endswith(b'\n'):
        raise ValueError('incomplete journal')
    previous = None
    result = []
    for line in raw.splitlines():
        if not line:
            raise ValueError('empty journal line')
        obj = json.loads(line)
        if canonical(obj) != line or set(obj) != {'index', 'previous', 'kind', 'payload', 'sha256'}:
            raise ValueError('noncanonical event')
        body = {k: obj[k] for k in ('index', 'previous', 'kind', 'payload')}
        if obj['index'] != len(result) or obj['previous'] != previous or obj['sha256'] != digest(body):
            raise ValueError('journal chain mismatch')
        result.append(obj)
        previous = obj['sha256']
    return result


class Journal:
    """Exclusive durable append with a caller-set bound; fixture seam only.

    A complete writer must also account for PNGs, report, completion, filesystem
    floor, wall time, and final failure reserve. This class cannot authorize use.
    """
    def __init__(self, path, limit):
        if type(limit) is not int or limit < 0:
            raise ValueError('invalid bound')
        self.path = Path(path)
        self.limit = limit
        self.stream = self.path.open('xb')
        self.index = 0
        self.previous = None
        self.size = 0
        self.fault = False
        fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def emit(self, kind, payload):
        if self.fault:
            raise ValueError('journal fault')
        body = dict(index=self.index, previous=self.previous, kind=kind, payload=payload)
        event = dict(body, sha256=digest(body))
        data = canonical(event) + b'\n'
        if self.size + len(data) > self.limit:
            self.fault = True
            raise ValueError('journal limit before append')
        try:
            if self.stream.write(data) != len(data):
                raise OSError('short journal write')
            self.stream.flush()
            os.fsync(self.stream.fileno())
        except BaseException:
            self.fault = True
            raise
        self.size += len(data)
        self.index += 1
        self.previous = event['sha256']
        return event

    def close(self):
        self.stream.close()
