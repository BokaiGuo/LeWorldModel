from copy import deepcopy
import hashlib

import numpy as np
import pytest

from experiments.spectral_reachability import p1_value_source_export_20261029 as m
from test_agent_p1_joint_vreach_saved_audit_20261029_v1 import paid, requests, inputs, complete


def fixed_inputs():
    prefix = np.zeros((5, 2), dtype='<f4')
    suffix = np.full((5, 2), .25, dtype='<f4')
    plan = dict(plan_digest='plan', prefix_commands=prefix.tolist(), suffix_commands=suffix.tolist())
    c = dict(original_plan=plan, full10_raw_bits=(prefix.tobytes() + suffix.tobytes()).hex())
    s = dict(index=0, task_id='task', anchor_id='anchor', context_sha256='a', goal_sha256='b',
        plan_digest='plan', job_digest='job', actual_return_event_sha256='event',
        prefix_sha256=hashlib.sha256(prefix.tobytes()).hexdigest(),
        suffix_sha256=hashlib.sha256(suffix.tobytes()).hexdigest(), predicted_Q=2.,
        actual_Q=3., physical_metrics={'native_distance': 4.})
    slots = [dict(s, index=i) for i in range(40)]
    return {'slots': slots}, {'candidates': [deepcopy(c) for _ in slots]}


def test_fixed_available_and_unknown():
    rebuilt, sealed = fixed_inputs(); rebuilt['slots'][12]['actual_Q'] = None
    rows = m.fixed_rows(rebuilt, sealed)
    assert len(rows) == 40 and rows[0]['suffix']['value'] == 2.
    assert rows[0]['physical']['value']['native_distance'] == 4.
    assert rows[12]['observed_terminal']['status'] == 'missing'
    assert all(r['immediate']['status'] == r['vreach']['status'] == 'missing' for r in rows)


def test_fixed_rejects_changed_action():
    rebuilt, sealed = fixed_inputs(); sealed['candidates'][3]['original_plan']['prefix_commands'][0][0] = -0.0
    with pytest.raises(ValueError, match='action bits'): m.fixed_rows(rebuilt, sealed)


def test_fixed_calls_full_rebuild_and_verification(monkeypatch):
    rebuilt, sealed = fixed_inputs(); calls = []
    monkeypatch.setattr(m.fixed, 'build', lambda: rebuilt)
    monkeypatch.setattr(m.fixed, 'verify', lambda path, sha, data: calls.append((sha, data)))
    class Reader:
        @staticmethod
        def read(*args): return sealed
    class Upstream:
        teacher = Reader(); SELECTOR = 's'; SELECTOR_SHA = 'h'
    monkeypatch.setattr(m.fixed, 'upstream', lambda: Upstream())
    assert len(m.export_fixed()['rows']) == 40
    assert calls == [(m.FIXED_SHA, rebuilt)]


def test_arm_must_pass_strict_audit_and_never_claims_physics(monkeypatch):
    q = {'slot': 0}
    prepared = {'authority': {'rows': [{'request': q}]}}
    candidate = dict(start=0, iteration=0, raw_float32_bits='00', mid={'x': 1},
                     stage_Q=1., terminal_Q=2., own_reach_value=3.)
    bundle = dict(slot=0, arm='a', result={'candidates': [candidate]*20}, scalar_receipt={}, events=[])
    calls = []
    monkeypatch.setattr(m.arm_audit, 'audit', lambda *a: calls.append(a) or {'candidates': 20})
    out = m.export_joint_arm(prepared, bundle)
    assert calls[0][0] is q and len(out['rows']) == 20
    assert out['rows'][0]['vreach']['value'] == 3.
    assert out['rows'][0]['physical']['status'] == 'missing'
    assert out['journal_origin'] == 'not authenticated by arm audit'
    def refuse(*args): raise ValueError('invalid audit')
    monkeypatch.setattr(m.arm_audit, 'audit', refuse)
    with pytest.raises(ValueError, match='invalid audit'): m.export_joint_arm(prepared, bundle)


def test_arm_actual_saved_consumer_with_ram_native_events(paid):
    q, result, receipt, events = paid
    prepared = {'authority': {'rows': [{'request': q}]}}
    bundle = dict(slot=0, arm=result['arm'], result=result, scalar_receipt=receipt, events=events)
    out = m.export_joint_arm(prepared, bundle)
    assert len(out['rows']) == 20
    assert out['rows'][0]['vreach']['value'] == result['candidates'][0]['own_reach_value']
    bundle['result'] = deepcopy(result)
    bundle['result']['candidates'][0]['stage_Q'] += 1
    with pytest.raises(ValueError): m.export_joint_arm(prepared, bundle)
