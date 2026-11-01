"""RAM metadata/array fixtures only; no model execution, GPU, or physics."""
from copy import deepcopy
import builtins
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_common_budget_20261028_v1 as m

# Reuse only the existing RAM fixtures; importing this file executes no tests
# or models. It is bound in the production source closure via native.own_files.
spec = importlib.util.spec_from_file_location('common_budget_ram_fixtures', m.ROOT/'tests/agent_p3_native_program_20261028_v1_test.py')
ram = importlib.util.module_from_spec(spec); spec.loader.exec_module(ram)


def fixture():
    old, source = ram.fixture()
    for b in source['decision_rows']:
        d = b['queue']['decision']; d.update(extension_type='progress', continue_expanding=True)
        for req in b['queue']['requests']:
            req.update(parent_context=deepcopy(old['tasks'][b['slot']]['root_context']))
    panels = {mode: deepcopy(source) for mode in m.MODES}
    branches = m.branches_from(old['tasks'], panels)
    for b in branches:
        b['signal_report_sha256'] = m.digest(b['mode'])
        b['signal_resources'] = {a: n//2 for a, n in m.signal.profile()['resource_caps'].items()}
        b['inclusive_signal_wall_ns'] = {'RAM_fixture': 1}
        b['signal_panel_wall_seconds'] = 1.
        b['signal_task_total_wall_seconds'] = None
    jobs = m.jobs_from(branches)
    return dict(tasks=old['tasks'], branches=branches, jobs=jobs, rule=m.rule(), profile=m.native.profile(jobs)), panels


def complete():
    p, _ = fixture(); sink = ram.Sink(); ledger = m.core.Ledger(p['profile']['resource_caps'], sink=sink.event)
    rows = m.census(p['jobs']); sink.event(dict(kind='denominator', rows=deepcopy(rows)))
    winners = m.panel(p, lambda s: ram.RamReplay(p['tasks'][s], sink), ledger, sink, rows)
    receipt = ledger.receipt()
    r = dict(rows=rows, winners=winners, resource_receipt=receipt, per_branch=m.accounting(p, receipt, rows))
    r.update(equal_execution_horizon=m.equal_execution_horizon(p['branches']), fixed_H1_cost_minimal_baseline=False,
        same_closed_loop_success_claim=False, comparison_scope=m.rule()['comparison_scope'])
    return p, r, sink.events


def test_actual_queue_matched_total_and_full_J_cost():
    p, r, events = complete(); m.validate_saved(p, r, events)
    assert len(p['jobs']) == len(r['rows']) == 32
    assert r['equal_execution_horizon'] is False and r['fixed_H1_cost_minimal_baseline'] is False
    assert [b['candidates'] for b in p['branches']] == [12, 4, 12, 4]
    assert r['resource_receipt']['completed']['predict_API'] == r['resource_receipt']['completed']['predict_samples'] == 96
    assert r['resource_receipt']['completed']['replay_calls'] == 64
    for b in r['per_branch']:
        assert b['total_resources']['predict_API'] == b['total_resources']['predict_samples'] == 32
        assert b['signal_resources']['jacobian_API'] == 2 and b['signal_resources']['VJP_directions'] == 384
        assert b['total_resources']['predict_context_tokens'] == 96
        assert b['search_resources']['array_SVD_calls'] == 1
        assert b['total_resources']['array_SVD_calls'] == 3
        assert b['raw_action_steps'] == 5*b['execution_horizon_packets'] == 5*b['horizon']
    assert all(w['executed_extension_depth'] == 1 and w['requested_depth'] == 3 and w['physical_release'] is False for w in r['winners'])


def test_explicit_search_requests_dont_rewrite_original_queue():
    p, panels = fixture()
    assert all(b['original_decision']['candidates'] == 4 for b in p['branches'])
    assert panels['fixed']['decision_rows'][0]['queue']['decision']['candidates'] == 4
    assert len(p['branches'][0]['requests']) == 12
    for b in p['branches']:
        for k, req in enumerate(b['requests']):
            assert req['operation'] == 'common_budget_root_progress' and req['candidate_index'] == k
            assert req['budget_candidates'] == b['candidates'] and req['prefix_edges'] == req['prefix_actions'] == []
            assert req['parent_context_sha256'] == m.digest(p['tasks'][b['slot']]['root_context'])


@pytest.mark.parametrize('fault', ['quota', 'child', 'reconfig', 'composition', 'missing_pair', 'cross_root', 'missing_branch', 'horizon'])
def test_queue_rejects_invalid_dependencies(fault):
    p, panels = fixture(); b = panels['fixed']['decision_rows'][0]
    if fault == 'quota': b['queue']['decision']['candidates'] = 5
    elif fault == 'child': b['queue']['depth'] = 1
    elif fault == 'reconfig': b['queue']['decision']['program_type'] = 'reconfiguration'
    elif fault == 'composition': b['queue']['decision']['program_type'] = 'composition'
    elif fault == 'missing_pair': panels['fixed']['rows'].pop(0)
    elif fault == 'cross_root': panels['fixed']['rows'][0]['response']['parent_context_sha256'] = 'a'*64
    elif fault == 'missing_branch': panels['fixed']['decision_rows'].pop(0)
    else: b['queue']['decision']['horizon'] = 2
    with pytest.raises(ValueError): m.branches_from(p['tasks'], panels)


def test_adaptive_H1_has_twelve_useful_candidates_no_padding():
    p, panels = fixture()
    for b in panels['adaptive']['decision_rows']:
        if b['mode'] == 'adaptive': b['queue']['decision']['horizon'] = 1
    branches = m.branches_from(p['tasks'], panels)
    assert [b['candidates'] for b in branches] == [12]*4
    assert m.equal_execution_horizon(branches) is True
    assert m.native.profile(m.jobs_from(branches))['resource_caps']['predict_API'] == 96


def test_generator_fixed_normalizer_Q_and_degenerate_collisions_kept():
    p, _ = fixture(); t = p['tasks'][0]; r = deepcopy(p['branches'][0]['response'])
    r['jacobian'] = np.zeros((192, 10), np.float32).tolist()
    ledger = m.core.Ledger(p['profile']['resource_caps']); ledger.scope = dict(test=True)
    bank = m.generate(t, r, 12, ledger)
    assert len(bank['actions']) == 12 and len({m.digest(a) for a in bank['actions']}) == 1
    assert bank['fractions'] == [i/11 for i in range(12)]
    assert ledger.completed['solve_calls'] == ledger.completed['array_SVD_calls'] == 1
    assert ledger.completed['predict_API'] == 0


@pytest.mark.parametrize('fault', ['actions', 'missing', 'cost', 'winner', 'padding', 'free_J', 'samples', 'API', 'SVD', 'raw', 'normalizer', 'module', 'terminal', 'missing_io', 'scope', 'outcome', 'decision', 'equal_horizon', 'cheap_fixed', 'closed_loop'])
def test_strict_saved_mutations_rejected(fault):
    p, r, events = complete()
    if fault == 'actions': r['rows'][0]['actions'][0][0] += .5
    elif fault == 'missing': r['rows'].pop()
    elif fault == 'cost': r['rows'][0]['cost'] += 1
    elif fault == 'winner': r['winners'][0]['index'] = 100
    elif fault == 'padding': r['per_branch'][0]['candidates'] += 1
    elif fault == 'free_J': p['branches'][0]['signal_resources']['predict_API'] = 0
    elif fault in ('samples', 'API', 'SVD'):
        axis = {'samples': 'predict_samples', 'API': 'predict_API', 'SVD': 'array_SVD_calls'}[fault]
        r['resource_receipt']['completed'][axis] -= 1
    elif fault == 'terminal': r['rows'][0]['whole'][-1]['latent_history'][-1][0] += 1
    elif fault == 'missing_io': events.pop(next(i for i, e in enumerate(events) if e['kind'] == 'io_return'))
    elif fault == 'scope': next(e for e in events if e['kind'] == 'io_return')['scope']['slot'] = 99
    elif fault == 'outcome': events[0]['outcome'] = 'leak'
    elif fault == 'decision': p['branches'][0]['candidates'] = 11
    elif fault == 'equal_horizon': r['equal_execution_horizon'] = True
    elif fault == 'cheap_fixed': r['fixed_H1_cost_minimal_baseline'] = True
    elif fault == 'closed_loop': r['same_closed_loop_success_claim'] = True
    else:
        kind, key = {'raw': ('program_packet', 'raw'), 'normalizer': ('program_packet', 'std'), 'module': ('io_return', 'pred_proj:output')}[fault]
        e = next(e for e in events if e['kind'] == kind and key in e['arrays'])
        a = m.signal.unblob(e['arrays'][key]).copy(); a.flat[0] += 1; e['arrays'][key] = m.signal.blob(a)
    with pytest.raises((ValueError, KeyError)): m.validate_saved(p, r, events)


def test_paid_failure_preserves_all_slots_and_no_refund():
    p, _ = fixture(); rows = m.census(p['jobs']); sink = ram.Sink()
    ledger = m.core.Ledger(p['profile']['resource_caps'], sink=sink.event)
    with pytest.raises(RuntimeError): m.panel(p, lambda s: ram.RamReplay(p['tasks'][s], sink, fail=True), ledger, sink, rows)
    assert len(rows) == 32 and rows[0]['status'] == 'failed'
    assert all(r['cost'] is None for r in rows)
    assert ledger.attempted['action_encoder_calls'] == 1 and ledger.completed['action_encoder_calls'] == 0
    assert ledger.attempted['solve_calls'] == ledger.completed['solve_calls'] == 1


def test_budget_denies_before_native_dispatch():
    p, _ = fixture(); caps = p['profile']['resource_caps'].copy(); caps['solve_calls'] = 0
    ledger = m.core.Ledger(caps); ledger.scope = dict(test=True)
    with pytest.raises(m.core.BudgetExceeded): m.generate(p['tasks'][0], p['branches'][0]['response'], 12, ledger)
    assert not any(ledger.attempted.values())


@pytest.mark.parametrize('fault', ['same_output', 'same_report', 'same_journal', 'tasks', 'model', 'missing', 'none'])
def test_two_independent_acquisitions_required(monkeypatch, tmp_path, fault):
    tasks = fixture()[0]['tasks']; outputs = {mode: tmp_path/mode for mode in m.MODES}
    inputs = {mode: dict(output=str(outputs[mode]), report_sha256=m.digest(mode)) for mode in m.MODES}
    panels = {mode: dict(prepared=mode, prepare_sha256=m.digest(mode+'prepare'), artifact_sha256={'events.jsonl': m.digest(mode+'journal')}) for mode in m.MODES}
    prepared = {mode: dict(tasks=deepcopy(tasks), policy='original', expected_model_hash='a'*64, runtime_rule='original') for mode in m.MODES}
    if fault == 'same_output': inputs['adaptive']['output'] = inputs['fixed']['output']
    elif fault == 'same_report': inputs['adaptive']['report_sha256'] = inputs['fixed']['report_sha256']
    elif fault == 'same_journal': panels['adaptive']['artifact_sha256'] = deepcopy(panels['fixed']['artifact_sha256'])
    elif fault == 'tasks': prepared['adaptive']['tasks'][0]['goal'][0] += 1
    elif fault == 'model': prepared['adaptive']['expected_model_hash'] = 'b'*64
    elif fault == 'missing': inputs.pop('adaptive')
    monkeypatch.setattr(m.signal, 'consume', lambda output, expected: dict(resources={}))
    monkeypatch.setattr(m.signal, 'read', lambda path: panels[Path(path).parent.name])
    monkeypatch.setattr(m.signal, 'verify', lambda path, expected: prepared[path])
    if fault == 'none':
        ps, rs, receipts = m.load_panels(inputs)
        assert set(ps) == set(rs) == set(receipts) == set(m.MODES)
    else:
        with pytest.raises(ValueError): m.load_panels(inputs)


def test_import_and_unauthorized_run_no_new_neural_import(monkeypatch):
    old = builtins.__import__; attempts = []
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'gymnasium', 'pymunk'): attempts.append(name); raise AssertionError('forbidden import')
        return old(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    with pytest.raises(ValueError, match='HOST CUDA'): m.run(None, None, None, 'not_authorized')
    assert not attempts


def test_cold_plan_no_model_import():
    script = 'import sys; from experiments.spectral_reachability import agent_p3_common_budget_20261028_v1 as m; m.main(["plan"]); assert not any(n.split(".")[0] in ("torch","gymnasium","pymunk") for n in sys.modules)'
    proc = subprocess.run([sys.executable, '-B', '-c', script], cwd=m.ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr


def test_metadata_verify_roundtrip_and_rule_tamper(tmp_path, monkeypatch):
    p, _ = fixture(); p.update(inputs={'fixed': {}, 'adaptive': {}})
    path = tmp_path/'prepare.json'; path.write_bytes(m.canonical(p)); expected = m.sha(path)
    monkeypatch.setattr(m, 'material', lambda inputs: deepcopy(p))
    assert m.verify(path, expected) == p
    p['rule']['total_predict_per_task_method'] += 1
    with pytest.raises(ValueError): m.verify(path, expected)


def test_saved_consume_roundtrip(tmp_path, monkeypatch):
    p, r, events = complete(); output = tmp_path/'saved'; output.mkdir()
    p.update(sources={}, expected_model_hash='d'*64, historical_acquisition={'unknown': None})
    prepared = str(tmp_path/'prepare.json'); prep_sha = 'a'*64; hashes = dict(original_state_dict='d'*64, all_buffers='b'*64)
    events += [dict(kind='model_before', hashes=hashes), dict(kind='model_after', hashes=hashes), dict(kind='sources_final', sha256=m.digest({}))]
    tail = '0'*64; raw = []
    for i, payload in enumerate(events):
        e = dict(index=i, previous=tail, payload=payload); tail = m.digest(e); raw.append(m.canonical(dict(e, sha256=tail))+b'\n')
    (output/'events.jsonl').write_bytes(b''.join(raw))
    (output/'binding.json').write_bytes(m.canonical(dict(prepared=prepared, prepare_sha256=prep_sha)))
    r.update(schema=m.NAME+'.report', status='equal_predict_saved_not_admitted', prepared=prepared, prepare_sha256=prep_sha,
        errors=[], sources_verified=True, source_manifest_sha256=m.digest({}), model_before=hashes, model_after=hashes,
        saved_verification=dict(upstream_saved_SVD_calls=16, semantics='two upstream panels *4 SVD *pre/post verify; partial work unknown'),
        host={'RAM_fixture_not_CUDA': True}, wall_seconds=1., CUDA_peak_bytes=[0, 0], journal_tail=tail,
        artifact_sha256={n: m.sha(output/n) for n in ('binding.json', 'events.jsonl')}, **m.FLAGS)
    (output/'report.json').write_bytes(m.canonical(r)); monkeypatch.setattr(m, 'verify', lambda path, expected: p)
    result = m.consume(output, m.sha(output/'report.json'))
    assert result['denominator'] == 32 and result['scientific_effect'] is result['physical_benefit'] is None
    assert result['equal_execution_horizon'] is result['fixed_H1_cost_minimal_baseline'] is result['same_closed_loop_success_claim'] is False
    assert all(b['total_resources']['predict_API'] == b['total_resources']['predict_samples'] == 32 for b in result['per_branch'])


def test_frozen_predecessors_and_admission():
    assert m.sha(m.native.__file__) == 'c616c95f463b07bd0356c3855aba96ec7d0fbc492060ab4304854d7be790f44f'
    assert m.sha(m.signal.__file__) == '6ce7722eca40cdd5f8958ba9590d8dc3d76d38bd10895d5ca494c1db3b6d100b'
    assert all(value is False for value in m.FLAGS.values())


def test_material_real_wiring_with_explicit_ram_input_verifier(tmp_path, monkeypatch):
    # Test material() itself, not only a fabricated prepared dict. Replace only
    # upstream verification with explicit RAM metadata; no model or old file edit.
    p, panels = fixture(); tasks = p['tasks']; inputs = {}; prepared = {}; receipts = {}
    half = {a: n//2 for a, n in m.signal.profile()['resource_caps'].items()}
    for mode in m.MODES:
        output = tmp_path/mode; output.mkdir(); prepared_path = tmp_path/(mode+'_prepare.json')
        prepared_path.write_bytes(b'RAM upstream input, not evidence')
        inputs[mode] = dict(output=str(output), report_sha256=m.digest(mode))
        for name in ('report.json', 'events.jsonl', 'binding.json'): (output/name).write_bytes(b'RAM metadata, not evidence')
        panels[mode].update(prepared=str(prepared_path), wall_seconds=1., resource_receipt=dict(events=[
            dict(scope=dict(slot=slot), status='returned', delta=half, resource='RAM_signal_fixture', started_ns=0, ended_ns=1) for slot in (0, 1)]))
        prepared[mode] = dict(tasks=tasks, policy='RAM policy', expected_model_hash='a'*64, sources={}, history={'unmeasured_acquisition': None})
        receipts[mode] = dict(resources=m.signal.profile()['resource_caps'])
    previous = tmp_path/'RAM_predecessor'; previous.write_bytes(b'RAM lineage, not evidence')
    monkeypatch.setattr(m, 'PREVIOUS', previous); monkeypatch.setattr(m, 'PREVIOUS_SHA', m.sha(previous))
    monkeypatch.setattr(m, 'load_panels', lambda inputs: (prepared, panels, receipts))
    out = m.material(inputs)
    assert len(out['jobs']) == 32 and out['profile']['resource_caps']['predict_API'] == 96
    assert sum(b['signal_resources']['predict_API'] for b in out['branches']) == 32
    assert all(b['signal_task_total_wall_seconds'] is None for b in out['branches'])
    assert out['predecessor_outcomes_read'] is False
    assert all(m.sha(path) == expected for path, expected in out['sources'].items())
