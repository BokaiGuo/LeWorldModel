import copy

import pytest
from experiments.spectral_reachability import p3_native_cert_prep_20261029 as m
from experiments.spectral_reachability import agent_p3_c3_k17_native_outer_20261029_v1 as outer


def fixture():
    actions = [[f'{i * 5 + j:064x}' for j in range(5)] for i in range(6)]
    caps = dict.fromkeys(m.AXES, 6)
    events = []
    for i, identity in enumerate(m.ORDER):
        events += [{'kind': 'open', 'identity': identity},
                   {'kind': 'charge', 'axis': 'predict', 'amount': 1},
                   {'kind': 'guard', 'accepted': True, 'history_sha256': 'a'*64,
                    'goal_sha256': 'b'*64, 'replay_sha256': 'c'*64}]
        for j, action in enumerate(actions[i]):
            events += [{'kind': 'intent', 'index': j, 'action_sha256': action},
                       {'kind': 'return', 'index': j, 'terminated': False}]
        events.append({'kind': 'close'})
    return events, caps, actions


def test_complete_is_only_fixture_and_never_authority():
    events, caps, actions = fixture()
    result = m.audit_fixture(events, caps, actions)
    assert m.ORDER == outer.ORDER
    assert result['transcript_complete'] and result['completed_slots'] == 6
    assert not any(result[k] for k in ('dispatch_allowed', 'scientific_pass', 'budget_authorized', 'native_provenance_verified'))


@pytest.mark.parametrize('mutation', ['guard', 'action', 'cap', 'order', 'return', 'boolean'])
def test_rejects_invalid_transcripts(mutation):
    events, caps, actions = fixture()
    if mutation == 'guard': events[2]['accepted'] = False
    if mutation == 'action': events[3]['action_sha256'] = 'f'*64
    if mutation == 'cap': caps['predict'] = 0
    if mutation == 'order': events[0]['identity'] = m.ORDER[1]
    if mutation == 'return': events[4]['index'] = 3
    if mutation == 'boolean': caps['predict'] = True
    with pytest.raises(ValueError): m.audit_fixture(events, caps, actions)


def test_failure_and_pending_are_never_completion():
    events, caps, actions = fixture()
    partial = copy.deepcopy(events[:4])
    result = m.audit_fixture(partial, caps, actions)
    assert result['pending_action'] == 0 and not result['transcript_complete']
    partial.append({'kind': 'failure'})
    result = m.audit_fixture(partial, caps, actions)
    assert result['fail_stopped'] and result['pending_action'] == 0
    with pytest.raises(ValueError): m.audit_fixture(partial + [{'kind': 'close'}], caps, actions)


def test_native_termination_is_fail_stop():
    events, caps, actions = fixture()
    events[4]['terminated'] = True
    result = m.audit_fixture(events[:5], caps, actions)
    assert result['fail_stopped'] and not result['transcript_complete']
    with pytest.raises(ValueError): m.audit_fixture(events, caps, actions)
