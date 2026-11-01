"""Fixed extra-candidate selection seam; callbacks are fixture-only, not a runner."""
from __future__ import annotations
import hashlib
import json
import math
from copy import deepcopy

ARMS = ('zero_extra', 'witness_extra')
SLOTS = tuple((r, a) for r in (0, 1) for a in ARMS)
BASE_PACKET_CALLS = 144
EXTRA_PACKET_CALLS = 2
REPLAY_PACKET_CALLS = 4
PACKETS_PER_SLOT = 150


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def raw_actions(raw, n):
    if not isinstance(raw, list) or len(raw) != n:
        raise ValueError('raw action count')
    for action in raw:
        if not isinstance(action, list) or len(action) != 2:
            raise ValueError('raw action shape')
        if any(type(x) not in (int, float) or not math.isfinite(x) or not -1 <= x <= 1 for x in action):
            raise ValueError('raw action domain')
    return deepcopy(raw)


def extras(prefix):
    prefix = raw_actions(prefix, 5)
    zeros = [[0.0, 0.0] for _ in range(5)]
    return {'zero_extra': deepcopy(zeros + zeros), 'witness_extra': prefix + zeros}


class Meter:
    def __init__(self):
        self.attempted = 0
        self.returned = 0
        self.failed = False

    def invoke(self, packets, callback):
        if self.failed or packets not in (1, 2) or self.attempted + packets > 6:
            raise ValueError('extra/replay cap or poisoned meter')
        self.attempted += packets
        try:
            value = callback()
            if not isinstance(value, list) or len(value) != packets:
                raise ValueError('incomplete packet return')
            self.returned += packets
            return value
        except BaseException:
            self.failed = True
            raise


def select_and_replay(bank, expected_bank_sha, arm, witness_prefix, dispatch, score, meter):
    """Select from frozen original bank + one candidate; never update CEM.

    dispatch(raw, phase, prior_context) returns one context per packet. score
    implements the original target/goal Q and must be source-bound by caller.
    Bank costs and callback results here are not authenticated production IO.
    """
    if arm not in ARMS or not isinstance(bank, list) or len(bank) != 72 or digest(bank) != expected_bank_sha:
        raise ValueError('sealed 72-candidate bank required')
    for i, row in enumerate(bank):
        if row.get('index') != i or type(row.get('cost')) not in (int, float) or not math.isfinite(row['cost']):
            raise ValueError('candidate order or cost')
        raw_actions(row.get('commands'), 10)
    extra = extras(witness_prefix)[arm]
    evaluated = meter.invoke(2, lambda: dispatch(deepcopy(extra), 'extra', None))
    try:
        extra_cost = score(deepcopy(evaluated))
        if type(extra_cost) not in (int, float) or not math.isfinite(extra_cost):
            raise ValueError('invalid extra Q')
    except BaseException:
        meter.failed = True
        raise
    winner = min(range(73), key=lambda i: ((bank[i]['cost'] if i < 72 else extra_cost), i))
    selected = raw_actions(bank[winner]['commands'] if winner < 72 else extra, 10)
    whole = meter.invoke(2, lambda: dispatch(deepcopy(selected), 'whole', None))
    first = meter.invoke(1, lambda: dispatch(deepcopy(selected[:5]), 'segment_first', None))
    second = meter.invoke(1, lambda: dispatch(deepcopy(selected[5:]), 'segment_second', deepcopy(first[0])))
    return dict(arm=arm, bank_sha256=expected_bank_sha, extra=extra, extra_cost=extra_cost,
                extra_contexts=evaluated, winner_index=winner, selected_raw=selected,
                whole=whole, segmented=first + second, attempted=meter.attempted,
                returned=meter.returned, production_authorized=False)
