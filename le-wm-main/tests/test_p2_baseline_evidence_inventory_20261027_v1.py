"""Small arithmetic/JSON fixtures only: no neural, simulator, network or real results."""
from copy import deepcopy
import json

import pytest

from experiments.spectral_reachability import p2_baseline_evidence_inventory_20261027_v1 as m


def fixture():
    identity = {'goal_rgb_sha256': 'g', 'history_rgb_sha256': 'h', 'state': [1, 2]}
    budget = dict(planning_predict_samples=80, replay_predict_samples=4,
                  compilation_predict_samples=0, total_predict_samples=84, jacobian_calls=None)
    comparison, physical = {'rows': []}, {'rows': []}
    for root in ('a' * 64, 'b' * 64):
        cell = dict(root_sha256=root, physical_identity=deepcopy(identity),
                    object_pose_cost=3.0, official_state_distance=4.0, official_success=0,
                    predicted_q_mse=1.0, real_encoder_q_mse=2.0, budget=deepcopy(budget),
                    metric_sha256='c' * 64, goal_vectors_sha256='d' * 64,
                    action_domain={'units': 'raw'}, backend_policy='unknown')
        comparison['rows'].append(dict(root_sha256=root, arms={'cem': cell}))
        physical['rows'].append(dict(root_sha256=root, method='cem',
            context_identity=deepcopy(identity), full_horizon_completed=True,
            planned_commands=10, completed_commands=10, pose={'pose_cost': 3.0},
            official={'official_state_distance': 4.0, 'official_success': False}))
    return comparison, physical


def test_stable_root_join_independent_of_array_order():
    comp, phys = fixture()
    original = m.local_rows(comp, phys, 'cem')
    phys['rows'].reverse()
    assert m.local_rows(comp, phys, 'cem') == original
    assert original[0]['budget']['jacobian_calls'] is None
    assert original[0]['metrics']['object_pose_cost'] == 3


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'duplicate', 'wrong_arm', 'wrong_root',
                                      'context', 'partial', 'steps', 'pose', 'distance', 'success', 'bad_label'])
def test_fail_closed_local_evidence(mutation):
    comp, phys = fixture()
    row = phys['rows'][0]
    if mutation == 'missing':
        phys['rows'].pop()
    elif mutation == 'extra':
        extra = deepcopy(row)
        extra['root_sha256'] = 'e' * 64
        phys['rows'].append(extra)
    elif mutation == 'duplicate':
        phys['rows'].append(deepcopy(row))
    elif mutation == 'wrong_arm':
        row['method'] = 'other'
    elif mutation == 'wrong_root':
        comp['rows'][0]['arms']['cem']['root_sha256'] = 'e' * 64
    elif mutation == 'context':
        row['context_identity']['state'] = [9]
    elif mutation == 'partial':
        row['full_horizon_completed'] = False
    elif mutation == 'steps':
        row['completed_commands'] = 9
    elif mutation == 'pose':
        row['pose']['pose_cost'] = 30
    elif mutation == 'distance':
        row['official']['official_state_distance'] = 40
    elif mutation == 'success':
        row['official']['official_success'] = True
    elif mutation == 'bad_label':
        comp['rows'][0]['arms']['cem']['official_success'] = 2
    with pytest.raises(ValueError):
        m.local_rows(comp, phys, 'cem')


@pytest.mark.parametrize('value', [float('nan'), float('inf'), True, None, '3'])
def test_nonfinite_and_non_numeric_metrics(value):
    with pytest.raises(ValueError):
        m.finite(value)


@pytest.mark.parametrize('value', [-1, 85, True, None, '84'])
def test_invalid_budget(value):
    budget = fixture()[0]['rows'][0]['arms']['cem']['budget']
    budget['total_predict_samples'] = value
    with pytest.raises(ValueError):
        m.budget_check(budget)


def test_compilation_work_not_erased():
    budget = dict(planning_predict_samples=80, replay_predict_samples=4,
                  compilation_predict_samples=4, total_predict_samples=88)
    assert m.budget_check(budget)['total_predict_samples'] == 88
    budget['total_predict_samples'] = 84
    with pytest.raises(ValueError):
        m.budget_check(budget)


@pytest.mark.parametrize('raw', ['{"x":NaN}', '{"x":1,"x":2}'])
def test_strict_json(tmp_path, raw):
    (tmp_path / 'input.json').write_text(raw)
    with pytest.raises(ValueError):
        m.Reader(tmp_path).read('input.json')


def test_reader_bounded_and_no_escape(tmp_path, monkeypatch):
    (tmp_path / 'input.json').write_text(json.dumps({'x': 1}))
    reader = m.Reader(tmp_path)
    assert reader.read('input.json') == {'x': 1}
    assert reader.receipts['input.json']['bytes'] == 8
    with pytest.raises(ValueError, match='escapes'):
        reader.raw('../outside.json')
    monkeypatch.setattr(m, 'CAP', 4)
    with pytest.raises(ValueError, match='cap'):
        reader.raw('input.json')


def test_reported_tamper_fails_without_original_sources(tmp_path):
    export = tmp_path / m.EXPORT
    export.mkdir(parents=True)
    (export / 'report.json').write_text('{}')
    (export / 'comparison_readiness.json').write_text('{}')
    (export / 'SHA256SUMS').write_text('0' * 64 + '  ' + m.EXPORT + 'report.json\n')
    with pytest.raises(ValueError, match='SHA mismatch'):
        m.reported(m.Reader(tmp_path))


def test_import_boundary_is_stdlib_and_metadata_only():
    import ast
    tree = ast.parse(m.Path(m.__file__).read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not any(any(x in name for x in ('torch', 'numpy', 'gym', 'requests', 'subprocess'))
                   for name in imports if name)
