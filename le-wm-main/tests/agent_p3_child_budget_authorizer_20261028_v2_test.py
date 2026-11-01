"""RAM/saved-only regression tests for the v2 budget authorizer."""
from copy import deepcopy

import pytest

from experiments.spectral_reachability import agent_p3_child_budget_authorizer_20261028_v2 as m


@pytest.fixture(scope='module')
def saved_prepare():
    return m.producer.base.read(
        m.ROOT / 'results/agent_p3_child_program_cuda_20261028_v1_prepare_MAIN01.json',
        64 * 1024**2)


def test_expected_cap_vector_is_canonical_and_every_axis_is_bound():
    expected = m.inputs_template()
    assert m._cap_identity(expected) == m.CAP_VECTOR_SHA256
    for field in ('common_node_cap', 'common_method_cap'):
        for axis in m.AXES:
            tampered = deepcopy(expected)
            tampered[field][axis] -= 1
            with pytest.raises(ValueError):
                m.limits(tampered)


def test_non_cap_contract_fields_are_bound_too():
    tampered = deepcopy(m.inputs_template())
    tampered['signal_sharing'] = 'shared'
    with pytest.raises(ValueError):
        m.limits(tampered)


def test_failed_ticket_rejects_strategy_switch_and_keeps_attempt(saved_prepare):
    gate = m.WholeNodeGate(m.inputs_template(), saved_prepare['children'],
                           saved_prepare['tasks'])
    child = saved_prepare['children'][0]
    key = gate.open(child['child_id'], 'fixed', child['context_sha256'],
                    child['ancestry']['task_binding_sha256'])['ticket']
    with pytest.raises(RuntimeError):
        gate.call(key, 'predict', lambda: (_ for _ in ()).throw(RuntimeError('RAM fault')),
                  shape=[1, 3, 192])
    switched = gate.open(child['child_id'], 'adaptive', child['context_sha256'],
                         child['ancestry']['task_binding_sha256'])
    assert switched['status'] == 'precondition_refused'
    assert switched['reason'] == 'fail_stop'
    assert switched['failure_ticket']['ticket'] == key
    assert gate.actual['fixed']['attempted']['predict_API'] == 1
    assert gate.actual['fixed']['completed']['predict_API'] == 0
    assert gate.pending and gate.failed


def test_failure_receipt_replays_without_releasing_or_retrying(saved_prepare):
    gate = m.WholeNodeGate(m.inputs_template(), saved_prepare['children'],
                           saved_prepare['tasks'])
    child = saved_prepare['children'][0]
    key = gate.open(child['child_id'], 'fixed', child['context_sha256'],
                    child['ancestry']['task_binding_sha256'])['ticket']
    with pytest.raises(ValueError):
        gate.call(key, 'predict', lambda: (_ for _ in ()).throw(ValueError('bad')),
                  shape=[1, 3, 192])
    gate.open(child['child_id'], 'adaptive', child['context_sha256'],
              child['ancestry']['task_binding_sha256'])
    receipt = gate.receipt()
    checked = m.consume_gate(receipt)
    assert checked['status'].startswith('budget_hooks_replayed_')
    assert receipt['failure_ticket']['operation_id'] == 0


def test_tampered_receipt_cap_identity_rejected(saved_prepare):
    gate = m.WholeNodeGate(m.inputs_template(), saved_prepare['children'],
                           saved_prepare['tasks'])
    receipt = gate.receipt()
    bad = deepcopy(receipt)
    bad['limits']['common_node_cap']['predict_API'] -= 1
    with pytest.raises(ValueError):
        m.consume_gate(bad)


def test_runtime_is_still_fail_closed_and_neural_free():
    source = open(m.__file__, encoding='utf-8').read()
    assert 'import torch' not in source and 'import gymnasium' not in source
    out = m.runtime_preconditions()
    assert out['dispatch_allowed'] is False
    assert out['neural_calls'] == out['physics_calls'] == 0
