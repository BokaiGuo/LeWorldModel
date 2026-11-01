"""Lightweight RAM contracts only. No model, GPU, physics, or outcome asset."""
from copy import deepcopy
import builtins
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_fixed_execution_contract_20261028_v1 as m

spec = importlib.util.spec_from_file_location('fixed_execution_ram', m.ROOT/'tests/agent_p3_common_budget_20261028_v1_test.py')
previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
ram = previous.ram


class RamBackend(ram.RamReplay):
    kind = 'RAM_FIXTURE'


def prepared():
    base, _ = previous.fixture()
    return dict(base=base, contract=m.contract(), profile=m.profile(base['profile']), queue=m.queue(base))


def complete():
    p = prepared(); sink = ram.Sink(); ledger = m.core.Ledger(p['profile']['resource_caps'], sink=sink.event)
    rows = m.native.census(p['base']['jobs'])
    result = m.execute_panel(p, lambda s: RamBackend(p['base']['tasks'][s], sink), ledger, sink, rows, permit_ram=True)
    return p, result, sink.events


@pytest.fixture(scope='module')
def saved_ram(): return complete()


def test_fixed_execution_and_joint_planning_are_different(saved_ram):
    p, r, events = saved_ram
    assert r['status'] == 'fixed_execution_model_ready_physical_blocked', r['error']
    assert len(r['rows']) == 32 and len(r['prefixes']) == 4 and len(r['winners']) == 4
    assert [q['planning_horizon_packets'] for q in p['queue']] == [1, 3, 1, 3]
    assert all(q['requested_execution_steps'] == 5 and q['actual_child_expansion_depth'] == 0 for q in p['queue'])
    assert all(q['requested_tree_depth'] == 3 for q in p['queue'])
    assert r['equal_actual_physical_execution_horizon'] is None and r['actual_physics_status'] == 'unknown_not_run'
    assert events[0]['kind'] == 'fixed_execution_denominator'
    for prefix in r['prefixes']:
        request = prefix['request']; winner = next(w for w in r['winners'] if (w['slot'], w['mode']) == (prefix['slot'], prefix['mode']))
        assert request['commands'] == winner['actions'][:5] and len(request['commands']) == 5
        assert request['planned_action_steps'] in (5, 15)
        assert request['nonexecuted_suffix_steps'] == request['planned_action_steps']-5
        assert request['physical_result'] is request['physical_root_binding'] is request['physical_goal_binding'] is None
        assert request['physical_release'] is request['execution_authorized'] is False


def test_feasible_real_budget_includes_full_J_and_useful_prefix(saved_ram):
    p, r, _ = saved_ram; branches = m.audit_budget(p, r['resource_receipt'])
    assert r['resource_receipt']['completed']['predict_API'] == r['resource_receipt']['completed']['predict_samples'] == 100
    assert r['resource_receipt']['completed']['replay_calls'] == 68
    assert r['resource_receipt']['completed']['metric_calls'] == 36
    for b in branches:
        assert b['signal_resources']['predict_API'] == b['signal_resources']['predict_samples'] == 8
        assert b['signal_resources']['jacobian_API'] == 2 and b['signal_resources']['VJP_directions'] == 384
        assert b['new_model_resources']['predict_API'] == b['new_model_resources']['predict_samples'] == 25
        assert b['total_resources']['predict_API'] == b['total_resources']['predict_samples'] == 33
        assert b['actual_physics_steps'] is b['FLOPs'] is b['CUDA_ns'] is None


def test_ram_requires_explicit_opt_in_and_empty_rows():
    p = prepared(); sink = ram.Sink(); ledger = m.core.Ledger(p['profile']['resource_caps'], sink=sink.event)
    rows = m.native.census(p['base']['jobs'])
    with pytest.raises(ValueError, match='HOST CUDA'): m.execute_panel(p, lambda s: RamBackend(p['base']['tasks'][s], sink), ledger, sink, rows)
    assert not ledger.events
    rows[0]['status'] = 'returned'
    with pytest.raises(ValueError, match='denominator'):
        m.execute_panel(p, lambda s: RamBackend(p['base']['tasks'][s], sink), ledger, sink, rows, permit_ram=True)


@pytest.mark.parametrize('fault', ['extended', 'shortened', 'bool_H', 'cross_task', 'endpoint', 'bad_raw', 'dtype', 'missing_path'])
def test_selected_prefix_rejects_invented_or_replaced_actions(saved_ram, fault):
    p, r, _ = saved_ram; request = r['prefixes'][1]['request']; winner = deepcopy(r['winners'][1])
    original = next(row for row in r['rows'] if (row['slot'], row['mode'], row['index']) == (winner['slot'], winner['mode'], winner['index']))
    path = deepcopy(original['whole'][:1]); winner['prefix_context_sha256'] = m.digest(path[0]); task = p['base']['tasks'][winner['slot']]
    assert request['planning_horizon_packets'] == 3
    if fault == 'extended': winner['actions'].extend(winner['actions'][:5])
    elif fault == 'shortened': winner['actions'] = winner['actions'][:5]
    elif fault == 'bool_H': winner['horizon'] = True
    elif fault == 'cross_task': task = p['base']['tasks'][1-winner['slot']]
    elif fault == 'endpoint': winner['prefix_context_sha256'] = 'a'*64
    elif fault == 'bad_raw': winner['actions'][0][0] = 2.
    elif fault == 'dtype': winner['actions'][0][0] = .1234567890123
    else: path = []
    with pytest.raises(ValueError): m.execution_request(task, winner, path)


def test_signal_for_child_and_reconfiguration_blocked():
    p = prepared(); bad = deepcopy(p['base']); bad['branches'][0]['original_decision']['program_type'] = 'reconfiguration'
    with pytest.raises(ValueError): m.queue(bad)
    assert m.contract()['child_replanning'] is False
    assert m.contract()['no_root_signal_for_child'] is True


@pytest.mark.parametrize('fault', ['bool_H', 'child_prefix', 'stale_task', 'unpaid_J', 'missing_request'])
def test_prepare_queue_rejects_contract_mutations(fault):
    p = prepared(); b = p['base']['branches'][0]
    if fault == 'bool_H': b['horizon'] = True
    elif fault == 'child_prefix': b['requests'][0]['prefix_actions'] = [[0., 0.]]*5
    elif fault == 'stale_task': b['requests'][0]['binding_sha256'] = 'a'*64
    elif fault == 'unpaid_J': b['signal_resources']['jacobian_API'] = 0
    else: b['requests'].pop()
    with pytest.raises(ValueError): m.queue(p['base'])


def test_full_failure_denominator_and_no_refunds():
    p = prepared(); sink = ram.Sink(); ledger = m.core.Ledger(p['profile']['resource_caps'], sink=sink.event)
    result = m.execute_panel(p, lambda s: RamBackend(p['base']['tasks'][s], sink, fail=True), ledger, sink, m.native.census(p['base']['jobs']), permit_ram=True)
    assert result['status'] == 'failed_or_incomplete'
    assert len(result['rows']) == 32 and len(result['prefixes']) == 4
    assert result['rows'][0]['status'] == 'failed' and all(r['cost'] is None for r in result['rows'])
    assert all(r['status'] == 'blocked_not_dispatched' and r['request'] is None for r in result['prefixes'])
    assert result['resource_receipt']['attempted']['action_encoder_calls'] == 1
    assert result['resource_receipt']['completed']['action_encoder_calls'] == 0


def test_prefix_failure_does_not_replace_or_pad_winner():
    p = prepared(); sink = ram.Sink(); ledger = m.core.Ledger(p['profile']['resource_caps'], sink=sink.event)
    class Fault(RamBackend):
        def replay(self, parent, actions, segmented, ledger):
            if ledger.scope.get('operation') == 'selected_prefix_verification': raise RuntimeError('RAM prefix verification fault')
            return super().replay(parent, actions, segmented, ledger)
    result = m.execute_panel(p, lambda s: Fault(p['base']['tasks'][s], sink), ledger, sink, m.native.census(p['base']['jobs']), permit_ram=True)
    assert result['status'] == 'failed_or_incomplete' and len(result['winners']) == 4
    assert result['prefixes'][0]['status'] == 'failed'
    assert all(r['request'] is None for r in result['prefixes'])
    assert result['resource_receipt']['attempted']['replay_calls'] == 65
    assert result['resource_receipt']['completed']['replay_calls'] == 64


@pytest.mark.parametrize('axis', ['predict_API', 'predict_samples', 'replay_calls', 'metric_calls'])
def test_budget_receipt_mutations_rejected(saved_ram, axis):
    p, result, _ = saved_ram; r = deepcopy(result['resource_receipt']); r['completed'][axis] += 1
    with pytest.raises(ValueError): m.audit_budget(p, r)


def physical_fixture(saved_ram, n=5, stop='completed'):
    req = deepcopy(saved_ram[1]['prefixes'][0]['request'])
    return req, dict(request_sha256=m.digest(req), commands=req['commands'][:n], states=np.ones((n+1, 7)).tolist(),
        returned_steps=n, stop_reason=stop, native_success=None, error=None)


def test_physics_unknown_is_not_zero_and_native_early_stop(saved_ram):
    req, r = physical_fixture(saved_ram); v = m.audit_physical_receipt(req, r)
    assert v['full_length'] is True and v['native_success'] is None and v['physical_release'] is False
    req, r = physical_fixture(saved_ram, 2, 'terminated'); v = m.audit_physical_receipt(req, r)
    assert v['actual_returned_steps'] == 2 and v['eligible_for_complete_length_comparison'] is False
    assert v['status'] == 'native_early_stop' and v['artifact_authenticity_verified'] is False


@pytest.mark.parametrize('fault', ['overrun', 'padding', 'state5D', 'missing_state', 'missing_native', 'pretend_complete', 'wrong_binding', 'fake_release', 'error_suppressed'])
def test_physical_contract_fail_closed(saved_ram, fault):
    req, r = physical_fixture(saved_ram)
    if fault == 'overrun': r['returned_steps'] = 15
    elif fault == 'padding': r['commands'].append(req['commands'][0])
    elif fault == 'state5D': r['states'] = np.ones((6, 5)).tolist()
    elif fault == 'missing_state': r['states'].pop()
    elif fault == 'missing_native': r.pop('native_success')
    elif fault == 'pretend_complete': r['returned_steps'] = 2; r['commands'] = r['commands'][:2]; r['states'] = r['states'][:3]
    elif fault == 'wrong_binding': r['request_sha256'] = 'a'*64
    elif fault == 'fake_release': req['physical_release'] = True
    else: r['stop_reason'] = 'engineering_failure'
    with pytest.raises(ValueError): m.audit_physical_receipt(req, r)


def test_material_verify_queue_roundtrip_ram(tmp_path, monkeypatch):
    base = prepared()['base']; base.update(sources={}, rule=m.common.rule())
    src = tmp_path/'base.json'; src.write_bytes(m.canonical(base)); sha = m.sha(src)
    monkeypatch.setattr(m.common, 'verify', lambda path, expected: deepcopy(base))
    out = m.material(src, sha); assert len(out['queue']) == 4
    assert out['contract']['requested_execution_steps'] == 5 and out['profile']['resource_caps']['predict_API'] == 100
    path = tmp_path/'new.json'; path.write_bytes(m.canonical(out)); expected = m.sha(path)
    assert m.verify(path, expected) == out
    base['branches'][0]['candidates'] += 1
    with pytest.raises(ValueError): m.verify(path, expected)


def test_cold_plan_no_model_import():
    script = 'import sys; from experiments.spectral_reachability import agent_p3_fixed_execution_contract_20261028_v1 as m; m.main(["plan"]); assert not any(n.split(".")[0] in ("torch","pymunk","gymnasium") for n in sys.modules)'
    proc = subprocess.run([sys.executable, '-B', '-c', script], cwd=m.ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr


def test_frozen_inputs_and_science_closed():
    assert m.sha(m.common.__file__) == '5e2ea6e9644fa46e8387cf66baab19c7a001da06d2ee04e6bc60dc15fb4ec92a'
    assert m.contract()['equal_actual_physical_execution_horizon'] is None
    assert all(v is False for v in m.FLAGS.values())
