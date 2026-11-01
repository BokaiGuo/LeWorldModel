"""RAM arithmetic, negative mutations and read-only saved integration; no model."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p2_native_composition_gap_20261028_v1 as m


def test_native_squared_closure_and_units():
    x = m.decompose([3, 4, 0, 12, .5, 6, 8], [0] * 7)
    assert [x[k] for k in m.COMPONENTS] == [25., 144., .25, 100.]
    assert sum(x[k] for k in m.COMPONENTS) == x['native7_sq'] == 269.25
    assert x['native7'] ** 2 == pytest.approx(269.25)
    assert x['jointXY'] == 13


def test_native_angle_is_not_replaced_by_wrapping():
    x = m.decompose([0, 0, 0, 0, 2 * np.pi + .5, 0, 0], [0] * 7)
    assert x['native_angle_error'] == -.5  # exact installed evaluator, even outside usual domain
    assert x['wrapped_angle_diagnostic'] == pytest.approx(.5)
    assert x['raw_angle_sq'] == (2 * np.pi + .5) ** 2


@pytest.mark.parametrize('value', [None, [0]*5, [0]*6+[None], [0]*6+[True],
                                  [0]*6+['0'], [0]*6+[float('nan')], [0]*6+[float('inf')]])
def test_invalid_missing_state_never_becomes_zero(value):
    with pytest.raises(ValueError):
        m.decompose(value, [0]*7)


def ram_rows():
    rows = []
    for i in range(4):
        for a, state in zip(m.ARMS, ([10, 0, 1, 0, 0, 0, 0], [1, 0, 2, 0, 0, 20, 0])):
            rows.append(dict(root_slot=i, root_sha256=str(i), arm=a, goal=[0]*7,
                             initial_state=[0]*7, observed=m.decompose(state, [0]*7),
                             prediction=dict(Q_cost=1. if a == m.ARMS[0] else .5, physical_teacher_Q=None)))
    return rows


def test_ram_agent_object_velocity_reversals_and_missing_teacher():
    pairs = m.pair_rows(ram_rows())
    for p in pairs:
        assert p['lower_is_better']['objectXY'] == m.ARMS[0]
        assert p['lower_is_better']['jointXY'] == m.ARMS[1]
        assert p['lower_is_better']['native7'] == m.ARMS[0]
        assert p['composition_minus_joint']['teacher_Q'] is None
        assert p['predicted_vs_teacher_rank_disagreement'] is None
        assert sum(p['native7_squared_contributions'].values()) == p['native7_squared_difference']


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'bool_root', 'crossroot', 'goal', 'initial'])
def test_ram_pair_denominator_and_identity_rejection(mutation):
    rows = ram_rows()
    if mutation == 'missing': rows.pop()
    if mutation == 'duplicate': rows[-1] = deepcopy(rows[-2])
    if mutation == 'bool_root': rows[0]['root_slot'] = False
    if mutation == 'crossroot': rows[1]['root_sha256'] = 'wrong'
    if mutation == 'goal': rows[1]['goal'][0] = 1
    if mutation == 'initial': rows[1]['initial_state'][0] = 1
    with pytest.raises(ValueError): m.pair_rows(rows)


def test_literal_q_not_identity_default():
    context = dict(latent_history=np.zeros((3, 192)).tolist())
    goal = np.ones(192).tolist()
    assert m.latent_cost(context, goal, (2*np.eye(192)).tolist()) == 4
    with pytest.raises(ValueError): m.latent_cost(context, goal, None)


@pytest.fixture(scope='module')
def saved():
    docs = {k: json.loads((m.ROOT / p).read_text()) for k, (p, _) in m.INPUTS.items()}
    protocol = json.loads((m.ROOT / 'results/p2_composition_predict_frontier_cuda_20261027_v2_prepared01/protocol.json').read_text())
    comp = json.loads(Path(protocol['context']['compilation']).read_text())
    predictions = [json.loads((m.ROOT / f'results/agent_p2_runtime_guard_20261028_v1_known_MAIN01/root_{i}.json').read_text())
                   for i in range(4)]
    return (docs['physical'], docs['strict'], docs['audit'], docs['prepare'], predictions,
            {x['root_sha256']: x['compiled'] for x in comp['rows']}, docs['teacher'], docs['teacher_protocol'])


def test_real_saved_all_four_no_new_inference():
    r = m.load_saved()
    assert r['root_denominator'] == 4 and len(r['rows']) == 8
    assert r['actual_selected_step_returns'] == 80
    assert r['native_successes'] == dict.fromkeys(m.ARMS, 0)
    assert [r['composition_distance_wins'][k] for k in ('predicted_Q', 'teacher_Q', 'native7', 'jointXY', 'objectXY')] == [2, 1, 2, 3, 1]
    assert r['new_neural_calls'] == r['new_physics_steps'] == 0
    assert r['scientific_admission'] is r['P1_admission'] is r['P2_admission'] is False
    assert all(x['prediction']['physical_teacher_Q'] is not None for x in r['rows'])
    assert r['teacher_reuse']['samples'] == 8
    assert r['pairs'][2]['native7_squared_contributions']['agent_velocity_sq'] > 72000


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'state', 'goal', 'actions', 'selection',
                                      'cost', 'Q', 'audit', 'native_success', 'strict', 'metric', 'step'])
def test_saved_mutations_failclosed(saved, mutation):
    p, s, a, prepared, predictions, c, _, _ = deepcopy(saved)
    b = p['rows'][0]['physical']
    if mutation == 'missing': p['rows'].pop()
    if mutation == 'duplicate': p['rows'][-1] = deepcopy(p['rows'][0])
    if mutation == 'state': b['final_state'][0] += 1
    if mutation == 'goal': b['context_identity']['goal'][0] += 1
    if mutation == 'actions': prepared['plans'][0]['execution_plan']['commands'][0][0] = .125
    if mutation == 'selection': prepared['plans'][0]['source_selection_digest'] = '0'*64
    if mutation == 'cost': predictions[0]['results'][0]['winner']['cost'] += 1
    if mutation == 'Q': c[predictions[0]['root_sha256']]['scoring']['metric']['projection'][0][0] += 1
    if mutation == 'audit': a['rows'][0]['native7'] += 1
    if mutation == 'native_success': b['official']['official_success'] = True
    if mutation == 'strict': s['scientific_admission'] = True
    if mutation == 'metric': b['active_metric_id'] = 'wrong'
    if mutation == 'step': b['trace'][0]['primitive_step'] = 2
    with pytest.raises(ValueError): m.analyze_bundle(p, s, a, prepared, predictions, c)


@pytest.mark.parametrize('mutation', ['missing', 'image', 'state', 'context', 'goal', 'Q', 'model',
                                      'teacher_cost', 'prediction', 'vector', 'old_selection'])
def test_teacher_reuse_requires_exact_identity(saved, mutation):
    p, s, a, prepared, predictions, c, t, proto = deepcopy(saved)
    result = m.analyze_bundle(p, s, a, prepared, predictions, c)
    e = t['encodings'][0]
    if mutation == 'missing': t['encodings'].pop()
    if mutation == 'image': e['terminal_rgb_sha256'] = 'wrong'
    if mutation == 'state': t['rows'][0]['final_state'][0] += 1
    if mutation == 'context': proto['plans'][0]['predictor_context']['latent_history'][0][0] += 1
    if mutation == 'goal': proto['root_metrics'][e['root_sha256']]['goal'][0] += 1
    if mutation == 'Q': proto['root_metrics'][e['root_sha256']]['metrics']['source_Q']['projection'][0][0] += 1
    if mutation == 'model': proto['model_bindings'] = {}
    if mutation == 'teacher_cost': e['actual_costs']['source_Q'] += 1
    if mutation == 'prediction': e['predicted_costs']['source_Q'] += 1
    if mutation == 'vector': e['terminal_vector'][0] = None
    if mutation == 'old_selection': e['selection_sha256'] = 'wrong'
    with pytest.raises(ValueError): m.attach_teacher(result, p, prepared, c, t, proto)


def test_external_sha_is_not_optional(monkeypatch):
    monkeypatch.setattr(m, 'INPUTS', {**m.INPUTS, 'physical': (m.INPUTS['physical'][0], '0'*64)})
    with pytest.raises(ValueError, match='SHA drift'): m.load_saved()


def test_fresh_process_import_guard_and_saved_cli():
    # Global sys.modules may contain torch from unrelated tests. Never erase it.
    code = '''
import importlib.abc, sys
blocked = {'torch', 'gymnasium', 'stable_worldmodel', 'pymunk', 'requests'}
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, NoRuntime())
from experiments.spectral_reachability import agent_p2_native_composition_gap_20261028_v1 as m
assert not (blocked & sys.modules.keys())
result = m.load_saved()
assert result['new_neural_calls'] == result['new_physics_steps'] == 0
assert not (blocked & sys.modules.keys())
'''
    subprocess.run([sys.executable, '-B', '-c', code], cwd=m.ROOT,
                   env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'},
                   check=True, capture_output=True, text=True, timeout=30)


def test_cli_stdout_no_output_asset():
    proc = subprocess.run([sys.executable, '-B', '-m', 'experiments.spectral_reachability.' + m.NAME],
                          cwd=m.ROOT, check=True, capture_output=True, text=True, timeout=30,
                          env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'})
    assert json.loads(proc.stdout)['teacher_reuse']['samples'] == 8
