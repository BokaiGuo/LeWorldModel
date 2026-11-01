"""Offline identity bridge for externally authenticated P1 evidence; no model calls.

This checks transport consistency, not provenance or scientific admission.
"""
import argparse
import hashlib
import json
import math
import re
import struct


KINDS = ('immediate', 'suffix', 'vreach', 'physical')
COMMON = ('candidate_id', 'task_id', 'context_sha256', 'goal_sha256',
          'root_step', 'prefix_end_step', 'terminal_step', 'prefix_hex', 'suffix_hex')
SHA = re.compile(r'[0-9a-f]{64}\Z')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def action(value, label):
    require(type(value) is str and re.fullmatch(r'[0-9a-f]{80}', value) is not None,
            label + ' must be five 2-D little-endian float32 commands')
    data = bytes.fromhex(value)
    require(all(math.isfinite(x) and abs(x) <= 1 for x in struct.unpack('<10f', data)),
            label + ' nonfinite or out of bounds')
    return data


def join(records):
    """Join exactly four per-candidate receipts, leaving trust with their producers.

    Each producer must independently export the complete common identity and its
    own source receipt digest. Vreach is conditional on the prefix and parent,
    not a score of the executed suffix. SHA values are not authenticated here.
    """
    require(type(records) is list and len(records) == 4, 'exactly four receipts required')
    by_kind = {}
    for row in records:
        require(type(row) is dict and row.get('kind') in KINDS, 'unknown receipt')
        kind = row['kind']
        require(kind not in by_kind, 'duplicate receipt')
        require(all(k in row for k in COMMON), 'missing identity')
        for key in ('candidate_id', 'task_id'):
            require(type(row[key]) is str and bool(row[key]), 'missing ' + key)
        for key in ('context_sha256', 'goal_sha256', 'receipt_sha256'):
            require(type(row.get(key)) is str and SHA.fullmatch(row[key]) is not None, 'invalid ' + key)
        for key in ('root_step', 'prefix_end_step', 'terminal_step'):
            require(type(row[key]) is int, 'step must be an integer')
        require(row['root_step'] < row['prefix_end_step'] < row['terminal_step'], 'invalid time order')
        require(row['prefix_end_step'] - row['root_step'] == 5 and
                row['terminal_step'] - row['prefix_end_step'] == 5, 'expected two five-step packets')
        prefix = action(row['prefix_hex'], 'prefix'); suffix = action(row['suffix_hex'], 'suffix')
        require(type(row.get('full_action_sha256')) is str and
                row['full_action_sha256'] == hashlib.sha256(prefix + suffix).hexdigest(), 'full action digest')
        require(type(row.get('parent_context_sha256')) is str and
                SHA.fullmatch(row['parent_context_sha256']) is not None, 'invalid parent context')
        value = row.get('value')
        require(type(value) in (int, float) and math.isfinite(value), 'missing or invalid value')
        expected_step = row['prefix_end_step'] if kind in ('immediate', 'vreach') else row['terminal_step']
        require(type(row.get('value_step')) is int and row['value_step'] == expected_step, 'wrong value time')
        by_kind[kind] = row
    first = by_kind['immediate']
    for row in by_kind.values():
        for key in COMMON + ('full_action_sha256', 'parent_context_sha256'):
            require(type(first[key]) is type(row[key]) and first[key] == row[key], 'identity mismatch: ' + key)
    return {'identity': {k: first[k] for k in COMMON + ('full_action_sha256', 'parent_context_sha256')},
            'values': {k: by_kind[k]['value'] for k in KINDS},
            'receipt_sha256': {k: by_kind[k]['receipt_sha256'] for k in KINDS},
            'vreach_scope': 'prefix-conditioned, not suffix-conditioned',
            'source_authentication': 'not performed', 'scientific_admission': 'unchanged'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='JSON array of four independently exported receipts')
    args = parser.parse_args()
    with open(args.input, encoding='utf-8') as source:
        receipts = json.load(source)
    print(json.dumps(join(receipts), sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
