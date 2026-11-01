"""RAM/saved budget contract tests; never load torch, CUDA, or physics."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import builtins

import pytest

from experiments.spectral_reachability import agent_p3_child_budget_authorizer_20261028_v1 as m


@pytest.fixture(scope='module')
def saved_prepare():
    return m.producer.base.read(
        m.ROOT/'results/agent_p3_child_program_cuda_20261028_v1_prepare_MAIN01.json', 64*1024**2)


@pytest.fixture(scope='module')
def saved_inputs(saved_prepare):
    receipt = m.producer.base.read(m.STRICT, 128*1024**2)
    signals = m.producer.base.read(m.producer.SIGNAL_RECEIPT, 256*1024**2)
    return saved_prepare, receipt, signals


def test_quota_is_integer_and_overcap_is_atomic_refusal():
    assert m.quota(1, 4)['search_predict'] == 8
    assert m.quota(3, 4)['search_predict'] == 24
    assert m.quota(1, 8)['search_predict'] == 16
    q = m.quota(3, 8)
    assert q['search_predict'] == 48 and q['legal'] is False and q['cost'] is None
    with pytest.raises(ValueError): m.quota(2, 4)
    with pytest.raises(ValueError): m.quota(3, 5)


def test_common_budget_rounding_is_not_child_policy():
    # The older root comparison derives 12/4 from 24/(2H); child policy keeps
    # the frozen 4/8 candidate field and rejects H3x8 as one whole node.
    assert m.quota(1, 4)['cost']['predict_API'] == 9
    assert m.quota(3, 4)['cost']['predict_API'] == 25
    assert m.quota(3, 8)['legal'] is False
    assert 'round' not in m.inputs_template()['allocation']


def test_saved_full_plan_preserves_48_and_actual_fixed_adaptive_vector(saved_inputs):
    p, receipt, signals = saved_inputs
    plan = m.saved_plan(p, receipt, signals, m.inputs_template())
    assert len(plan['rows']) == 48
    assert plan['original_census'] == dict(pending=41, refused=5, not_reached=2)
    assert plan['same_cap'] is True
    assert plan['matched_historical_actual_vector'] is False
    assert plan['matched_historical_predict'] is False
    assert plan['all_observed_refused'] is False
    assert sum(r['status'] == 'whole_node_refused' for r in plan['rows']) == 5
    assert sum(r['status'] == 'budget_feasible_native_adapter_missing' for r in plan['rows']) == 41
    assert sum(r['status'] == 'not_reached' for r in plan['rows']) == 2
    assert all(r['CUDA_dispatch_allowed'] is False for r in plan['rows'])


def test_fixed_and_adaptive_have_identical_caps_but_distinct_actual_cost(saved_inputs):
    p, receipt, signals = saved_inputs
    plan = m.saved_plan(p, receipt, signals, m.inputs_template())
    assert plan['common_limits']['common_node_cap'] == m.inputs_template()['common_node_cap']
    fixed = plan['historical_actual']['fixed']['attempted']
    adaptive = plan['historical_actual']['adaptive']['attempted']
    assert fixed['predict_API'] != adaptive['predict_API']
    assert fixed['predict_samples'] != adaptive['predict_samples']
    assert plan['matched_historical_actual_vector'] is False


def test_limits_tamper_and_missing_dependency_are_machine_readable():
    limits = deepcopy(m.inputs_template())
    limits['common_node_cap']['predict_API'] -= 1
    with pytest.raises(ValueError): m.limits(limits)
    out = m.runtime_preconditions()
    assert out['status'] == 'precondition_refused'
    assert out['dispatch_allowed'] is False
    assert out['reason'] == 'missing_reviewed_budget_instrumented_child_native_adapter'
    assert out['missing'] and out['neural_calls'] == out['physics_calls'] == 0


def test_gate_enforces_fixed_48_order_and_context(saved_prepare):
    gate = m.WholeNodeGate(m.inputs_template(), saved_prepare['children'], saved_prepare['tasks'])
    child = saved_prepare['children'][0]
    opened = gate.open(child['child_id'], 'fixed', child['context_sha256'], child['ancestry']['task_binding_sha256'])
    assert opened['dispatch_allowed'] is True
    with pytest.raises(ValueError, match='fixed48slot order'):
        gate.open(saved_prepare['children'][1]['child_id'], 'adaptive',
                  saved_prepare['children'][1]['context_sha256'], saved_prepare['children'][1]['ancestry']['task_binding_sha256'])
    with pytest.raises(ValueError, match='same observed ancestry'):
        gate.open(child['child_id'], 'adaptive', '0'*64, child['ancestry']['task_binding_sha256'])


def test_gate_attempt_is_not_refunded_on_failure(saved_prepare):
    gate = m.WholeNodeGate(m.inputs_template(), saved_prepare['children'], saved_prepare['tasks'])
    child = saved_prepare['children'][0]
    key = gate.open(child['child_id'], 'fixed', child['context_sha256'], child['ancestry']['task_binding_sha256'])['ticket']
    with pytest.raises(RuntimeError):
        gate.call(key, 'predict', lambda: (_ for _ in ()).throw(RuntimeError('RAM fault')), shape=[1, 3, 192])
    assert gate.failed is True
    assert gate.actual['fixed']['attempted']['predict_API'] == 1
    assert gate.actual['fixed']['completed']['predict_API'] == 0
    assert gate.pending
    refused = gate.open(child['child_id'], 'adaptive', child['context_sha256'], child['ancestry']['task_binding_sha256'])
    assert refused['status'] == 'precondition_refused' and refused['reason'] == 'fail_stop'


def test_gate_rejects_unsupported_or_overcap_decision(saved_prepare, saved_inputs):
    _, _, signals = saved_inputs
    gate = m.WholeNodeGate(m.inputs_template(), saved_prepare['children'], saved_prepare['tasks'])
    child = saved_prepare['children'][0]
    key = gate.open(child['child_id'], 'fixed', child['context_sha256'], child['ancestry']['task_binding_sha256'])['ticket']
    # The refusal is tested at the pure quota layer; no fake paid signal is
    # accepted as evidence of a completed node.
    assert m.quota(3, 8)['refusal'] == 'original_whole_node_search_budget'
    bad = dict(horizon=3, candidates=4, extension_type='reconfiguration', program_type='reconfiguration', continue_expanding=True)
    fs = signals['rows'][0]['result']['acquisitions'][0]['signal']['scheduling_signal']
    with pytest.raises(ValueError): gate.seal(key, bad, fs)


def test_gate_receipt_is_chained_and_has_unknowns(saved_prepare):
    gate = m.WholeNodeGate(m.inputs_template(), saved_prepare['children'], saved_prepare['tasks'])
    child = saved_prepare['children'][0]
    gate.open(child['child_id'], 'fixed', child['context_sha256'], child['ancestry']['task_binding_sha256'])
    receipt = gate.receipt()
    assert receipt['fixed_denominator'] == receipt['child_denominator']*2 == 48
    assert receipt['neural_execution_certified'] is False
    assert receipt['matched_actual_vector'] is False
    assert receipt['FLOPs'] is receipt['cuda_ns'] is receipt['wall_ns'] is None
    assert receipt['events'][0]['previous_sha256'] is None
    assert receipt['events'][0]['event_sha256'] == m.digest({k:v for k,v in receipt['events'][0].items() if k != 'event_sha256'})


def test_import_and_preconditions_never_import_neural_modules(monkeypatch):
    source = Path(m.__file__).read_text()
    assert 'import torch' not in source and 'import gymnasium' not in source
    assert 'native_factory' not in source
    assert m.runtime_preconditions()['physics_calls'] == 0


def test_historical_consumer_is_required_before_saved_projection(monkeypatch, saved_inputs):
    p, receipt, signals = saved_inputs
    calls = []
    monkeypatch.setattr(m.producer, 'consume', lambda *a: calls.append(a) or receipt)
    # The saved projection itself is pure, but this checks the authorizer's
    # load_saved path has an explicit full producer consumer edge.
    m.load_saved()
    assert calls and calls[0][1] == m.REPORT_SHA


@pytest.mark.parametrize('axis', ['predict_API', 'predict_samples', 'array_SVD_calls', 'replay_calls'])
def test_common_cap_is_vector_not_predict_only(axis):
    limits = m.inputs_template()
    limits['common_node_cap'][axis] = max(0, limits['common_node_cap'][axis] - 1)
    with pytest.raises(ValueError): m.limits(limits)

