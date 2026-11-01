"""Bounded catalog descriptor projection, NOT coverage approval or resolver execution.

Only the supplied catalog is read. Evidence paths are opaque descriptors: no
stat, hash, open, glob, outcome decoding, physics or model imports for them.
"""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re

from experiments.spectral_reachability import p1_conditional_source_metadata_20261027_v1 as lexical

ROOT = Path(__file__).resolve().parents[2]
INPUT_SCHEMA = 'moving_historical_authority_inventory_20261027_v1.inventory'
OUTPUT_SCHEMA = 'root_exclusion_authority_resolution_20261027_v1.inventory'
INPUT_CAP = 2 * 1024**2
OUTPUT_CAP = 2 * 1024**2
ENTRY_CAP = 1024
FIELDS = ('path', 'sha256', 'family_hint')


def require(ok, message):
    if not ok:
        raise ValueError(message)


class CatalogProjection(lexical.Projection):
    """Reuse lexical skipping without modifying its global allowlist."""

    def __init__(self, raw):
        super().__init__(BytesIO(raw), INPUT_CAP)
        self.count = None

    def walk(self, path=()):
        self.whitespace()
        if path == () or (len(path) == 2 and path[0] == 'evidence_files'):
            require(self.get() == 123, 'catalog/descriptor object required')
            seen = set()
            self.whitespace()
            if self.peek() == 125:
                self.get()
                return
            while True:
                self.whitespace()
                key = self.string(True)
                require(key not in seen, 'duplicate metadata key')
                seen.add(key)
                require(len(seen) <= 128, 'metadata key cap')
                self.whitespace()
                require(self.get() == 58, 'JSON colon required')
                self.walk(path + (key,))
                self.whitespace()
                sep = self.get()
                if sep == 125:
                    break
                require(sep == 44, 'JSON object separator required')
        elif path == ('evidence_files',):
            require(self.get() == 91, 'evidence_files array required')
            self.count = 0
            self.whitespace()
            if self.peek() == 93:
                self.get()
                return
            while True:
                require(self.count < ENTRY_CAP, 'inventory entry cap')
                self.walk(path + (self.count,))
                self.count += 1
                self.whitespace()
                sep = self.get()
                if sep == 93:
                    break
                require(sep == 44, 'JSON array separator required')
        elif path == ('schema',) or (len(path) == 3 and path[0] == 'evidence_files' and path[2] in FIELDS):
            require(self.peek() == 34, 'literal descriptor string required')
            value = self.string(True)
            require(0 < len(value) <= 4096, 'nonempty bounded metadata string required')
            self.leaves[path] = value
        else:
            self.skip()


def convert(raw, expected_sha256):
    require(type(raw) is bytes and len(raw) <= INPUT_CAP, 'catalog byte cap')
    require(type(expected_sha256) is str and re.fullmatch('[0-9a-f]{64}', expected_sha256), 'input SHA256 required')
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, 'input SHA mismatch')
    p = CatalogProjection(raw)
    p.walk()
    p.whitespace()
    require(p.peek() is None, 'trailing JSON')
    require(p.leaves.get(('schema',)) == INPUT_SCHEMA, 'catalog schema')
    require(p.count is not None and 0 < p.count <= ENTRY_CAP, 'nonempty evidence_files required')
    files = []
    seen = set()
    for i in range(p.count):
        require(all(('evidence_files', i, k) in p.leaves for k in FIELDS), 'missing descriptor field')
        path, sha, family = (p.leaves['evidence_files', i, k] for k in FIELDS)
        candidate = Path(path)
        # Lexical checks only: never touch evidence, even if unavailable.
        require(candidate.is_absolute() and '..' not in candidate.parts and str(candidate) == path
                and candidate.is_relative_to(ROOT) and candidate != ROOT, 'literal project path required')
        require(re.fullmatch('[0-9a-f]{64}', sha), 'descriptor SHA256 required')
        require(path not in seen, 'duplicate/conflicting inventory path')
        seen.add(path)
        files.append(dict(path=path, sha256=sha, family=family))
    return dict(schema=OUTPUT_SCHEMA, files=files)


def write_new(path, payload):
    raw = (json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    require(len(raw) <= OUTPUT_CAP, 'output byte cap')
    # Exclusive creation also rejects dangling symlinks and concurrent creators.
    with Path(path).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    with Path(args.input).open('rb') as stream:
        raw = stream.read(INPUT_CAP + 1)
    payload = convert(raw, args.input_sha256)
    sha = write_new(args.output, payload)
    print(json.dumps(dict(path=str(Path(args.output).absolute()), sha256=sha,
                         input_sha256=args.input_sha256, files=len(payload['files']),
                         coverage_approved=False), sort_keys=True))


if __name__ == '__main__':
    main()
