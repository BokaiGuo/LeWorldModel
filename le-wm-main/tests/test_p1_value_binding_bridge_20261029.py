import copy
import hashlib
import struct

import pytest

from experiments.spectral_reachability import p1_value_binding_bridge_20261029 as bridge


def receipts():
    prefix = struct.pack('<10f', *([0.0] * 10))
    suffix = struct.pack('<10f', *([0.25] * 10))
    common = dict(candidate_id='slot-1', task_id='task-1', context_sha256='a' * 64,
                  parent_context_sha256='b' * 64, goal_sha256='c' * 64,
                  root_step=10, prefix_end_step=15, terminal_step=20,
                  prefix_hex=prefix.hex(), suffix_hex=suffix.hex(),
                  full_action_sha256=hashlib.sha256(prefix + suffix).hexdigest())
    return [dict(common, kind=k, value=1.0, receipt_sha256='d' * 64,
                 value_step=15 if k in ('immediate', 'vreach') else 20) for k in bridge.KINDS]


def test_complete_and_explicitly_untrusted():
    result = bridge.join(receipts())
    assert result['source_authentication'] == 'not performed'
    assert result['values'] == dict.fromkeys(bridge.KINDS, 1.0)


@pytest.mark.parametrize('key,value', [('candidate_id', 'other'), ('context_sha256', 'e' * 64),
    ('parent_context_sha256', 'e' * 64), ('goal_sha256', 'e' * 64), ('terminal_step', 21),
    ('value_step', 15), ('receipt_sha256', ''), ('value', float('nan'))])
def test_mismatch_fails(key, value):
    rows = receipts(); rows[3][key] = value
    with pytest.raises(ValueError):
        bridge.join(rows)


def test_signed_zero_and_action_hash():
    rows = receipts(); prefix = struct.pack('<10f', -0.0, *([0.0] * 9))
    rows[2]['prefix_hex'] = prefix.hex()
    rows[2]['full_action_sha256'] = hashlib.sha256(prefix + bytes.fromhex(rows[2]['suffix_hex'])).hexdigest()
    with pytest.raises(ValueError, match='identity mismatch'):
        bridge.join(rows)


def test_missing_duplicate_and_independent_records():
    rows = receipts()
    with pytest.raises(ValueError): bridge.join(rows[:3])
    altered = copy.deepcopy(rows); altered[3]['kind'] = 'suffix'
    with pytest.raises(ValueError): bridge.join(altered)
