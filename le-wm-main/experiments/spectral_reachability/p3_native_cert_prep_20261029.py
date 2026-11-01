"""Fixture-only event contract for future native P3 certification; no dispatch.

Input dictionaries are untrusted claims, not native observations. This module
does not read an environment, issue actions, approve a budget, or certify science.
"""
import re

ORDER = ((3, 0, 'fixedH1'), (4, 0, 'staticH3_progress'), (5, 0, 'adaptive'),
         (9, 1, 'fixedH1'), (10, 1, 'staticH3_progress'), (11, 1, 'adaptive'))
AXES = ('predict', 'sample', 'context', 'j', 'vjp', 'svd', 'schedule',
        'compile', 'solve', 'replay', 'metric')
HEX = re.compile(r'^[0-9a-f]{64}$')


def _hash(value):
    if not isinstance(value, str) or not HEX.fullmatch(value):
        raise ValueError('invalid digest')


def audit_fixture(events, caps, expected_actions):
    """Check a proposed ordered transcript, with caller-supplied fixture caps.

    No hashes or caps here authenticate their origin; this is not a receipt
    consumer. An incomplete, rejected, or failed transcript is never complete.
    """
    if set(caps) != set(AXES) or any(type(v) is not int or v < 0 for v in caps.values()):
        raise ValueError('invalid caps')
    if len(expected_actions) != 6 or any(len(x) != 5 for x in expected_actions):
        raise ValueError('expected six five-action sequences')
    for actions in expected_actions:
        for value in actions:
            _hash(value)
    spent = dict.fromkeys(AXES, 0)
    complete = 0
    active = False
    guarded = False
    pending = None
    count = 0
    stopped = False
    for e in events:
        if stopped:
            raise ValueError('event after fail-stop')
        kind = e.get('kind')
        if kind == 'open':
            if active or complete >= 6 or tuple(e.get('identity', ())) != ORDER[complete]:
                raise ValueError('slot order')
            active, guarded, pending, count = True, False, None, 0
        elif kind == 'charge':
            axis, amount = e.get('axis'), e.get('amount')
            if not active or pending is not None or axis not in spent or type(amount) is not int or amount <= 0:
                raise ValueError('invalid charge')
            spent[axis] += amount
            if spent[axis] > caps[axis]:
                raise ValueError('cap exceeded')
        elif kind == 'guard':
            if not active or guarded or pending is not None or count or e.get('accepted') is not True:
                raise ValueError('guard refused or out of order')
            for name in ('history_sha256', 'goal_sha256', 'replay_sha256'):
                _hash(e.get(name))
            guarded = True
        elif kind == 'intent':
            if not active or not guarded or pending is not None or count >= 5:
                raise ValueError('intent out of order')
            if e.get('index') != count or e.get('action_sha256') != expected_actions[complete][count]:
                raise ValueError('action mismatch')
            pending = count
        elif kind == 'return':
            if pending is None or e.get('index') != pending or type(e.get('terminated')) is not bool:
                raise ValueError('return without intent')
            count += 1
            pending = None
            if e['terminated']:
                stopped = True
        elif kind == 'close':
            if not active or pending is not None or count != 5:
                raise ValueError('incomplete slot')
            active = False
            complete += 1
        elif kind == 'failure':
            if not active:
                raise ValueError('failure outside slot')
            stopped = True
        else:
            raise ValueError('unknown event')
    return {'fixture_only': True, 'completed_slots': complete, 'fixed_denominator': 6,
            'attempted_resources': spent, 'pending_action': pending,
            'fail_stopped': stopped, 'transcript_complete': complete == 6 and not active and not stopped,
            'native_provenance_verified': False, 'budget_authorized': False,
            'dispatch_allowed': False, 'scientific_pass': False}
