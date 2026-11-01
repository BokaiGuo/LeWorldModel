"""Fixtures validate proposed arithmetic/transport only, never experiment evidence."""
from copy import deepcopy
from fractions import Fraction
import json
import subprocess
import sys

import pytest

from experiments.spectral_reachability import agent_p3_nongreedy_counterfactual_contract_20261029_v1 as m


def bank():
    return [dict(x, status='complete', reason=None, immediate=1., reach=1., terminal=1.,
                 edge=[[0., 0.] for _ in range(5*x['horizon'])],
                 suffix=[[0., 0.] for _ in range(5)],
                 source_pointer='/fixture/'+str(i)) for i, x in enumerate(m.roster())]


def ret(step, raw):
    return dict(terminated=False, truncated=False, native7=[float(step)]*7)


def test_all52_fixed_orders_and_ties():
    a = m.analyze_bank(bank())
    assert len(a['panels']) == 8
    assert sum(x['denominator'] for x in a['panels']) == 48
    for x in a['panels']:
        assert x['original_order'] == []
        assert x['shadows']['finite_terminal'] == list(range(x['denominator']+1))
        assert all(d['physical_benefit'] is None for d in x['rows'])
    assert a['source_authenticated'] is False and a['scientific_admission'] is False


def test_original_P3_order_is_immediate_not_reach():
    rows = bank()
    rows[0].update(immediate=3., reach=.1)
    rows[1].update(immediate=2., reach=.9)
    panel = m.analyze_bank(rows)['panels'][0]
    assert panel['original_order'] == [1, 0]
    assert panel['shadows']['R_only'][:2] == [0, 1]


def test_P1_existing_counterexample_not_new_weight_search():
    # Eq16/proof-only toy: beta1 rejects stage4 + reach4/41 vs1+1.
    rows = bank()
    rows[0].update(immediate=4., reach=float(Fraction(4, 41)), terminal=.1)
    p = m.analyze_bank(rows)['panels'][0]
    assert p['rows'][0]['eligible'] is True
    assert p['original_order'] == [0]  # actual P3 is NOT P1 sum selection
    assert p['shadows']['stage_plus_R_beta1'][-1] == 0
    assert p['shadows']['finite_terminal'][0] == 0
    assert Fraction(p['rows'][0]['stage_plus_R_delta']) > 0


def test_reach_rejection_can_coexist_with_better_model_terminal():
    rows = bank(); rows[0].update(immediate=2., reach=2., terminal=.5)
    x = m.analyze_bank(rows)['panels'][0]['rows'][0]
    assert x['reach_rejects'] and not x['stage_rejects'] and x['model_terminal_better']
    assert not x['eligible'] and x['physical_benefit'] is None


def test_gate_tie_and_subtraction_overflow_do_not_add_tolerance():
    rows = bank(); rows[0].update(immediate=1.0000000000000002, reach=.9999999999999999)
    assert m.analyze_bank(rows)['panels'][0]['original_order'] == [0]
    rows[0]['reach'] = 1.
    assert m.analyze_bank(rows)['panels'][0]['original_order'] == []


@pytest.mark.parametrize('field,value', [('edge', [[2., 0.]]*5), ('edge', [[.1, 0.]]*5),
                                      ('terminal', float('nan')), ('reach', -1.),
                                      ('source_pointer', 'unbound')])
def test_malformed_candidate_failclosed(field, value):
    rows = bank(); rows[0][field] = value
    with pytest.raises(ValueError): m.analyze_bank(rows)


def test_no_roster_substitution_or_partial_denominator():
    rows = bank()
    with pytest.raises(ValueError): m.analyze_bank(rows[:-1])
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError): m.analyze_bank(rows)


def test_declared_null_retained_and_control_missing_not_zero():
    rows = bank()
    for i in (0, 12):
        rows[i].update(status='unavailable', reason='source_degenerate', immediate=None,
                       reach=None, terminal=None, edge=None, suffix=None)
    out = m.analyze_bank(rows)
    assert out['panels'][0]['denominator'] == 8
    assert out['panels'][0]['status'] == 'blocked_control_unavailable'
    rows[12] = bank()[12]
    assert m.analyze_bank(rows)['panels'][0]['rows'][0]['eligible'] is None


def test_zero_positive_bit_identity():
    rows = bank(); rows[12]['edge'][0][0] = -0.
    with pytest.raises(ValueError, match='positive zero'): m.analyze_bank(rows)


def test_replay_integer_full_vector_and_no_new_derivatives():
    s = m.replay_schedule(); c = m.replay_caps()
    assert len(s) == 104 and c['predict'] == 312
    assert c['action_encoder'] == c['predictor'] == c['pred_proj'] == 312
    assert c['AE_tokens'] == c['predictor_context_tokens'] == 936
    assert c['metric'] == 208 and c['jacobian'] == c['VJP'] == 0
    assert c['FLOPs'] is None and c['scalar_backward'] is None
    assert {x['raw_steps'] for x in s} == {10, 20}
    assert [x['layout'] for x in s[:2]] == ['whole', 'packet_segmented']


def test_RAM_full104_no_padding_no_refund():
    ledger = m.ReplayRAM()
    for request in m.replay_schedule():
        ledger.attempt(request, lambda r: dict(predict=r['predict'], action_encoder=r['predict'],
            predictor=r['predict'], pred_proj=r['predict'], metric=2))
    assert ledger.reserved_predict == 312 and len(ledger.units) == 104
    with pytest.raises(ValueError): ledger.attempt(m.replay_schedule()[0], lambda r: {})


def test_midreplay_failure_has_unknown_actual_and_cannot_retry():
    ledger = m.ReplayRAM(); first = m.replay_schedule()[0]
    def fail(r): raise RuntimeError('mid-module')
    with pytest.raises(RuntimeError): ledger.attempt(first, fail)
    assert ledger.reserved_predict == 2 and ledger.units[0]['actual'] is None
    assert ledger.units[0]['status'] == 'failed_unknown_actual'
    with pytest.raises(ValueError): ledger.attempt(first, fail)


def test_incomplete_resource_return_not_mislabelled_complete():
    ledger = m.ReplayRAM()
    with pytest.raises(ValueError): ledger.attempt(m.replay_schedule()[0], lambda r: dict(predict=2))
    assert ledger.units[0]['status'] == 'failed_unknown_actual'


def test_tail_uniform10_fixed_packet_budget():
    jobs = m.tail_jobs(); c = m.tail_caps()
    assert len(jobs) == 10 and [x['index'] for x in jobs] == list(range(10))
    assert [x['endpoint'] for x in jobs] == [i for i in range(5) for _ in range(2)]
    assert all(x['tail_raw5'] == [[0., 0.]]*5 and x['terminal_checkpoint'] == 20 for x in jobs)
    assert c['public_step'] == 300 and c['physics_step'] == 3040 and c['explicit_render'] == 960
    assert c['public_step']-c['original_public_step'] == c['incremental_public_step'] == 50
    assert c['physics_step']-c['original_physics_step'] == 500
    assert c['explicit_render']-c['original_render'] == 150
    assert [(x['root'], x['role']) for x in jobs[::2]] == list(m.ENDPOINTS)


def test_scientific_flags_closed_and_agent_velocity_axis():
    spec = m.specification()
    assert all(spec[k] is False for k in m.FLAGS)
    assert spec['native_axes'][-2:] == ['agent_vx', 'agent_vy']
    assert spec['tail_seconds'] == .5 and spec['tail_endpoint_absolute'] == 30
    assert 'not assumed zero' in spec['goal_velocity']
    assert 'not velocity magnitude' in spec['agent_velocity_error']


def test_tail_full10_lifecycle_and_no_original_mutation():
    ledger = m.TailRAM(); events = []; jobs = m.tail_jobs(); before = deepcopy(jobs)
    results = [ledger.run(job=x, cp15_exact=True, stopped=False, sink=events.append, callback=ret) for x in jobs]
    assert ledger.attempts == ledger.completed == 50 and jobs == before
    assert all(x['cp20']['native7'] == [20.]*7 for x in results)
    assert len(events) == 110 and events == ledger.events
    for k in range(10): assert events[11*k]['kind'] == 'cp15_guard'


@pytest.mark.parametrize('stopped,cp15', [(True, True), (True, False)])
def test_native_early_stop_never_forced_to_tail(stopped, cp15):
    ledger = m.TailRAM()
    r = ledger.run(job=m.tail_jobs()[0], cp15_exact=cp15, stopped=stopped,
                   sink=lambda e: pytest.fail('no guard after stop'), callback=lambda *_: pytest.fail('no dispatch'))
    assert r['cp20'] is None and r['status'] == 'native_stopped' and ledger.attempts == 0


def test_cp15_mismatch_no_guard_no_intent_no_callback():
    ledger = m.TailRAM(); events = []
    with pytest.raises(ValueError): ledger.run(job=m.tail_jobs()[0], cp15_exact=False, stopped=False,
        sink=events.append, callback=lambda *_: pytest.fail('no callback'))
    assert ledger.attempts == 0 and events == []


def test_guard_persistence_failure_no_tail_intent():
    ledger = m.TailRAM(); attempted = []
    def fail(e): attempted.append(e); raise OSError('fsync failed')
    with pytest.raises(OSError): ledger.run(job=m.tail_jobs()[0], cp15_exact=True, stopped=False,
        sink=fail, callback=lambda *_: pytest.fail('no callback'))
    assert [x['kind'] for x in attempted] == ['cp15_guard']
    assert ledger.events == [] and ledger.attempts == 0


def test_midtail_callback_no_refund_and_earlystop_null():
    ledger = m.TailRAM(); events = []
    def fail(step, raw): raise RuntimeError('unknown callback')
    with pytest.raises(RuntimeError): ledger.run(job=m.tail_jobs()[0], cp15_exact=True,
        stopped=False, sink=events.append, callback=fail)
    assert ledger.attempts == 1 and ledger.completed == 0
    def stop(step, raw): return dict(ret(step, raw), terminated=True)
    r = m.TailRAM().run(job=m.tail_jobs()[0], cp15_exact=True, stopped=False, sink=events.append, callback=stop)
    assert r['cp20'] is None and r['status'] == 'native_stopped'


def test_tail_cannot_replace_length_or_endpoint():
    job = m.tail_jobs()[0]; job['tail_raw5'] *= 2
    with pytest.raises(ValueError): m.TailRAM().run(job=job, cp15_exact=True, stopped=False,
        sink=lambda _: None, callback=ret)


def test_full_temporary_disk_build_write_verify(tmp_path):
    p = tmp_path/'contract.json'
    module = 'experiments.spectral_reachability.'+m.NAME
    cmd = [sys.executable, '-B', '-m', module]
    subprocess.run(cmd+['build', '--output', str(p)], check=True, cwd=m.ROOT, capture_output=True)
    subprocess.run(cmd+['verify', '--output', str(p)], check=True, cwd=m.ROOT, capture_output=True)
    obj = json.loads(p.read_bytes())
    assert obj == m.specification() and obj['production_authorized'] is False
    assert obj['denominator']['logical_C'] == 4 and obj['denominator']['inherited_no_witness'] == 3
    assert subprocess.run(cmd+['build', '--output', str(p)], cwd=m.ROOT, capture_output=True).returncode != 0
    obj['tail_jobs'][0]['terminal_checkpoint'] = 25; p.write_bytes(m.encode(obj))
    assert subprocess.run(cmd+['verify', '--output', str(p)], cwd=m.ROOT, capture_output=True).returncode != 0


def test_source_mismatch_failclosed(monkeypatch, tmp_path):
    p = tmp_path/'bad.py'; p.write_text('different')
    monkeypatch.setattr(m, 'PINS', {str(p): '0'*64})
    with pytest.raises(ValueError, match='source pin'): m.specification()


def test_relative_cli_path_rejected():
    with pytest.raises(ValueError): m.absolute('relative.json')
