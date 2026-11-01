"""Saved/RAM regression only. Never verify through neural/physics execution."""
from collections import Counter
from copy import deepcopy
import inspect
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import agent_p3_child_native_physical_saved_20261028_v2 as m


@pytest.fixture(scope='module')
def saved():
    m.pins()
    output = m.ACTUAL_OUTPUT
    assert m.sha(output/'report.json') == m.ACTUAL_REPORT_SHA
    r = m.READ(output/'report.json', m.base.FINAL)
    assert m.sha(r['prepared']) == r['prepare_sha256'] == m.ACTUAL_PREPARE_SHA
    p = m.READ(r['prepared'], 64*1024**2)
    assert m.sha(output/'journal.jsonl') == r['journal_sha256']
    events = m.base.journal(output/'journal.jsonl', m.base.JOURNAL_CAP)
    return p, r, events, output


def test_exact_copy_of_full_frozen_validator_only_two_census_changes():
    before = inspect.getsource(m.base.validate_saved).strip()
    expected = before.replace('reset=3, set_state=6, goal_render=3, goal_setter=4',
                              'reset=2, set_state=4, goal_render=2, goal_setter=3')
    expected = expected.replace('physics_step=6+10*(10+b+n)', 'physics_step=4+10*(10+b+n)')
    assert expected != before
    assert inspect.getsource(m.validate_saved).strip() == expected


def test_real_saved_v1_exact_census_failure_preserved(saved):
    p, r, es, out = saved
    with pytest.raises(ValueError, match='census: reset'):
        m.base.validate_saved(p, r, es, out)


def test_real_saved_new_consumer_full_validate_41_5_2_and_earlystop(saved):
    p, r, es, out = saved
    result = m.validate_saved(p, r, es, out)
    assert result['complete'] is True and result['new_neural_calls'] == 0
    assert result['schema'] == m.NAME+'.consume'
    assert result['fixed_denominator'] == 48 and len(result['rows']) == 48
    assert Counter(x['status'] for x in result['rows']) == dict(returned5=37, native_stopped=4,
            blocked_whole_node_search_budget=5, not_reached=2)
    stops = [x for x in result['rows'] if x['status'] == 'native_stopped']
    assert [x['own_returned'] for x in stops] == [5, 2, 5, 1]
    assert sum(x['own_returned'] for x in result['rows']) == 198
    expected = dict(constructor=41, close=41, reset=82, set_state=164, goal_render=82,
                    goal_setter=123, goal_bind=41, public_step=993, physics_step=10094, explicit_render=1075)
    assert result['resources'] == {k:dict(attempted=v, completed=v) for k,v in expected.items()}
    assert all(result[k] is False for k in m.FLAGS)
    # The historical upper cap is left sealed; it is NOT relabeled actual cost.
    assert p['profile']['caps']['reset'] == 123
    assert p['profile']['caps']['physics_step'] == 10246


def test_duplicate_success_trajectory_is_not_independent_evidence(saved):
    _, r, _, _ = saved
    a = next(x for x in r['rows'] if x['episode_id'] == 'e01_I_b05_fixed')
    b = next(x for x in r['rows'] if x['episode_id'] == 'e01_R_b05_fixed')
    assert a['executed_raw'] == b['executed_raw']
    assert a['last_observation']['snapshot'] == b['last_observation']['snapshot']
    assert a['final_metrics']['native_success'] is b['final_metrics']['native_success'] is True


@pytest.mark.parametrize('resource', ['reset', 'set_state', 'goal_render', 'goal_setter', 'physics_step'])
def test_extra_operation_not_accepted_as_optional_constructor_cost(saved, monkeypatch, resource):
    p, r, es, out = saved; r = deepcopy(r); es = deepcopy(es)
    monkeypatch.setattr(m, 'png', lambda *a: None)
    ident = max(e['operation_id'] for e in es if 'operation_id' in e)+1
    fields = dict(operation_id=ident, episode_id=r['rows'][0]['episode_id'], resource=resource, units=1, region='setup')
    es.extend([dict(index=len(es), kind='operation_attempted', **fields), dict(index=len(es)+1, kind='operation_completed', **fields)])
    r['resources'][resource]['attempted'] += 1; r['resources'][resource]['completed'] += 1
    with pytest.raises(ValueError, match='census: '+resource): m.validate_saved(p, r, es, out)


@pytest.mark.parametrize('mutation', ['drop', 'raw', 'context', 'angle2pi', 'goal', 'metric', 'earlystop_padding', 'gate', 'public_cost'])
def test_other_original_validations_retained(saved, monkeypatch, mutation):
    p, r, es, out = saved; r = deepcopy(r); es = deepcopy(es)
    monkeypatch.setattr(m, 'png', lambda f, d: m.base.native_frame(f))
    if mutation == 'drop': r['rows'].pop()
    if mutation == 'raw': next(e for e in es if e['kind'] == 'step_return')['command'][0] = 0.
    if mutation == 'context': next(e for e in es if e['kind'] == 'child_boundary_sealed')['context_sha256'] = '0'*64
    if mutation == 'angle2pi': next(e for e in es if e['kind'] == 'fresh_observation')['receipt']['snapshot']['bodies']['block']['unwrapped_angle'] += 2*m.np.pi
    if mutation == 'goal': next(e for e in es if e['kind'] == 'fresh_observation')['receipt']['goal'][0] += 1.
    if mutation == 'metric': r['rows'][0]['final_metrics']['native7_reward'] += 1.
    if mutation == 'earlystop_padding': next(x for x in r['rows'] if x['status'] == 'native_stopped' and x['own_returned'] == 2)['own_returned'] = 5
    if mutation == 'gate': r['P3_admission'] = True
    if mutation == 'public_cost': r['resources']['public_step']['completed'] -= 1
    with pytest.raises((ValueError, KeyError)): m.validate_saved(p, r, es, out)


def test_saved_wrapper_uses_full_original_verify_without_old_consume(saved, monkeypatch):
    p, r, es, out = saved; calls = []
    real = m.base
    def verify(path, h):
        assert path == r['prepared'] and h == m.ACTUAL_PREPARE_SHA
        calls.append('full_base_verify_RAM_stub'); return p
    def forbidden(*a, **k): pytest.fail('old consume/neural/physics must never execute')
    # Replace ONLY this module's reference with a RAM namespace, never old globals.
    alias = SimpleNamespace(**vars(real)); alias.verify = verify; alias.consume = forbidden
    alias.run = forbidden; alias.run_verified = forbidden
    monkeypatch.setattr(m, 'base', alias)
    monkeypatch.setattr(m, 'validate_saved', lambda *a: dict(complete=True, rows=r['rows']))
    result = m.consume(out, m.ACTUAL_REPORT_SHA)
    assert calls == ['full_base_verify_RAM_stub']
    assert result['report_sha256'] == m.ACTUAL_REPORT_SHA
    assert result['repair']['physics_replayed'] is False
    assert set(m.PINS).issubset(result['repaired_consumer_sources'])


def test_external_wrong_sha_before_full_verify(monkeypatch):
    calls = []
    alias = SimpleNamespace(**vars(m.base)); alias.verify = lambda *a: calls.append('verify')
    monkeypatch.setattr(m, 'base', alias)
    with pytest.raises(ValueError, match='external actual'): m.consume(m.ACTUAL_OUTPUT, '0'*64)
    assert calls == []


def test_frozen_three_source_pins(saved):
    assert m.pins() == m.PINS
    source = inspect.getsource(m)
    assert 'monkeypatch' in m.__doc__
    assert '.native_factory(' not in source and '.run(' not in source
    assert '.verify(' in source and 'base.material' not in source
