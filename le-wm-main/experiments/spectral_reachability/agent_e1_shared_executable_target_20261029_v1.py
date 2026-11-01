"""Prospective target-only assembly seam; never authenticates reachability."""
from __future__ import annotations

import base64
import hashlib
import json
import math
import struct

SCHEMA = 'e1_shared_executable_target_fixture_v1'
ROOTS = (0, 1)


def _sha(s):
    if not isinstance(s, str) or len(s) != 64 or any(c not in '0123456789abcdef' for c in s):
        raise ValueError('expected lowercase sha256')
    return s


def assemble(records, interventions):
    """Construct isolated controller inputs; supplied provenance is NOT certified.

    records: fixed root-ordered mappings with target bytes (base64 little-endian
    f32[192]), task/history/goal identity and separate witness-reference hashes.
    interventions: preregistered distinct labels, not outcome-derived choices.
    The returned controller payload contains no witness or physical labels.
    """
    if not isinstance(interventions, list) or len(interventions) < 2 or any(
        not isinstance(x, str) or not x or len(x) > 80 for x in interventions
    ) or len(set(interventions)) != len(interventions):
        raise ValueError('at least two distinct intervention labels required')
    if not isinstance(records, list) or len(records) != 2:
        raise ValueError('exactly two preregistered root records required')
    rows = []
    for root, record in zip(ROOTS, records, strict=True):
        if set(record) != {'root', 'task_sha256', 'history_sha256', 'goal_sha256',
                           'target_b64', 'target_sha256', 'witness_sha256',
                           'witness_receipt_sha256'} or record['root'] != root:
            raise ValueError('root order or fields')
        for name in ('task_sha256', 'history_sha256', 'goal_sha256', 'target_sha256',
                     'witness_sha256', 'witness_receipt_sha256'):
            _sha(record[name])
        try:
            raw = base64.b64decode(record['target_b64'], validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError('target encoding') from exc
        if len(raw) != 768 or base64.b64encode(raw).decode() != record['target_b64']:
            raise ValueError('canonical f32[192] target bytes required')
        if hashlib.sha256(raw).hexdigest() != record['target_sha256']:
            raise ValueError('target digest mismatch')
        if not all(math.isfinite(x) for x in struct.unpack('<192f', raw)):
            raise ValueError('nonfinite target')
        for intervention in interventions:
            rows.append({'root': root, 'intervention': intervention,
                         'controller_input': {k: record[k] for k in (
                             'task_sha256', 'history_sha256', 'goal_sha256',
                             'target_b64', 'target_sha256')},
                         'witness_reference': {k: record[k] for k in (
                             'witness_sha256', 'witness_receipt_sha256')}})
    return {'schema': SCHEMA, 'rows': rows, 'provenance_authenticated': False,
            'physical_executability_authenticated': False,
            'production_authorized': False, 'scientific_gate': False}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()
