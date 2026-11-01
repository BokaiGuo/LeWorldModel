"""Offline arithmetic/negative-contract tests using the frozen saved cohort."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import p3_quality_cost_audit_20261027_v2 as audit

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def evidence():
    paths = [audit.MAIN,audit.BINDING,audit.PHYSICAL,
        'results/P3_ADAPTIVE_RUNTIME_20261027_cuda_v1/P3_ADAPTIVE_RUNTIME_report_20261027.json',
        'results/P3_ADAPTIVE_RUNTIME_20261027_cuda_v1/P3_ADAPTIVE_RUNTIME_protocol_20261027.json',
        'results/P3_MATCHED_PHYSICAL_20261027_v1_cuda/protocol.json']
    return [audit.decode((ROOT/p).read_bytes()) for p in paths]


def test_saved_census_and_dependence(evidence):
    result = audit.audit(*evidence)
    assert result['root_count'] == 4
    assert result['root_arm_count'] == 8
    assert len(result['rows']) == 16
    assert result['guard_aborted_executions'] == 4
    assert result['selected_path_physical_step_attempts_all_repeats'] == 154
    assert result['planning_predict_samples_once_per_root_arm'] == 1152
    assert result['verification_predict_samples_once_per_root_arm'] == 19
    assert result['total_saved_predict_samples_once_per_root_arm'] == 1171
    assert result['classifications_per_execution_pair'] == {
        'matched_full_path_saved_resources':2, 'matched_partial_guard_endpoint':2, 'unmatched':4}
    assert all(g['repeated_endpoints_equal'] for g in result['dependence_groups'])
    assert all(not result[k] for k in ('P1_admission','P3_admission','adaptive_superiority',
        'efficacy_claim','total_flops_equality','confidence_intervals_over_repeats'))
    partial = [r for r in result['rows'] if r['guard']['abstained']]
    assert all(r['actual_executed_primitive_steps']==6 and r['selected_horizon_tokens']==3
               and not r['full_horizon_completed'] and r['remaining_commands_not_executed']==9 for r in partial)
    assert all(r['jacobian_materialization_count'] is None and
               set(r['backward_and_vjp_counts'].values()) == {0} for r in result['rows'])


def test_order_independent_full_identity_join(evidence):
    expected = audit.audit(*evidence)
    for report, name in ((evidence[0],'corrections'),(evidence[1],'rows'),(evidence[2],'rows'),
                         (evidence[2],'encodings'),(evidence[3],'rows'),(evidence[5],'plans')):
        report[name].reverse()
    assert audit.audit(*evidence) == expected


@pytest.mark.parametrize('which,field',[(0,'corrections'),(1,'rows'),(2,'rows'),(2,'encodings'),(3,'rows'),(5,'plans')])
@pytest.mark.parametrize('mutation',['duplicate','missing','foreign'])
def test_reject_bad_join(evidence,which,field,mutation):
    rows = evidence[which][field]
    if mutation == 'duplicate':
        rows.append(deepcopy(rows[0]))
    elif mutation == 'missing':
        rows.pop()
    else:
        rows[0]['root_sha256'] = 'a'*64
    with pytest.raises(ValueError):
        audit.audit(*evidence)


@pytest.mark.parametrize('field,value',[
    ('selection_sha256','0'*64),('planned_horizon_tokens',99),
    ('corrected_search_horizon',99),('physical_env_step_returns',9),
    ('full_horizon_completed',False)])
def test_reject_misbound_physical(evidence,field,value):
    evidence[2]['rows'][0][field] = value
    with pytest.raises(ValueError):
        audit.audit(*evidence)


def test_reject_phase_counts(evidence):
    evidence[3]['rows'][0]['prediction_receipt']['phases'][0]['predict_samples'] += 1
    with pytest.raises(ValueError):
        audit.audit(*evidence)


def test_reject_plan_content_tamper(evidence):
    evidence[5]['plans'][0]['commands'][0][0] += .001
    with pytest.raises(ValueError,match='SHA'):
        audit.audit(*evidence)


@pytest.mark.parametrize('raw',[b'{"a":1,"a":2}',b'{"x":NaN}',b'{"x":1e999}'])
def test_strict_json(raw):
    with pytest.raises(ValueError):
        audit.decode(raw)


def test_reject_hash_change_and_symlink(tmp_path):
    evidence = audit.Evidence(tmp_path)
    path = tmp_path/'saved.json'
    path.write_bytes(b'{}')
    evidence.raw(path)
    with pytest.raises(ValueError,match='SHA256'):
        evidence.raw(path,'0'*64)
    link = tmp_path/'link.json'
    link.symlink_to(path)
    with pytest.raises(ValueError,match='symlink'):
        evidence.raw(link)
    path.write_bytes(b'{"changed":true}')
    with pytest.raises(ValueError,match='SHA256'):
        evidence.recheck()


def test_new_output_only(tmp_path):
    (tmp_path/'results'/audit.NAME).mkdir(parents=True)
    with pytest.raises(ValueError,match='exclusive NEW'):
        audit.run(tmp_path)


def test_no_neural_imports():
    import ast
    tree = ast.parse(Path(audit.__file__).read_text())
    names = [node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    names += [alias.name.split('.')[0] for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names]
    assert set(names) <= {'__future__','argparse','collections','hashlib','json','math','pathlib'}
