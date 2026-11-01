"""Non-neural adversarial tests; fixtures are not empirical efficacy evidence."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.spectral_reachability import p2_composition_admission_20261027_v1 as m


def calibration_rows():
    return [dict(root_sha256=str(i)*64, arm=arm, root_binding_sha256=str(i)*64,
                 fixed_horizon_complete=True, official_success=False,
                 predicted_q=1., object_pose_cost=2., official_state_distance=3.)
            for i in range(4) for arm in ('composition','flat_search','cem','random')]


def test_successful_sampling_is_never_admission():
    rows = calibration_rows()
    for row in rows:
        row.update(P1_admission=True, P2_admission=True, official_success=True,
                   fair_total_budget_comparison=True)
    result = m.decide(rows)
    assert result['evidence_integrity_valid']
    assert result['official_successes'] == 16
    assert result['P2_admission'] is False
    assert not any(result['necessary_scientific_gates'].values())
    assert all(result['executable_prerequisites'].values())


def test_root_join_is_order_invariant_and_keeps_guard():
    rows = calibration_rows()
    rows[0]['fixed_horizon_complete'] = False
    result = m.decide(rows[::-1])
    assert result['fixed_horizon_branches'] == 15
    assert result['selected_branches'] == 16
    assert result['independent_outcome_roots'] == 0


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'wrong_arm', 'root', 'goal_context'])
def test_identity_defects_reject(defect):
    rows = calibration_rows()
    if defect == 'missing': rows.pop()
    if defect == 'duplicate': rows[-1] = deepcopy(rows[0])
    if defect == 'wrong_arm': rows[-1]['arm'] = 'no_composition_sra'
    if defect == 'root': rows[-1]['root_sha256'] = 'x'*64
    if defect == 'goal_context': rows[-1]['root_binding_sha256'] = 'x'*64
    with pytest.raises(ValueError):
        m.decide(rows)


@pytest.mark.parametrize('raw', ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}'])
def test_strict_json(tmp_path, raw):
    path = tmp_path/'bad.json'
    path.write_text(raw)
    with pytest.raises(ValueError): m.read_json(path)


def test_hashes_and_concurrent_change(tmp_path):
    path = tmp_path/'input.json'
    path.write_text('{}')
    reader = m.EvidenceReader()
    reader.bind(path, m.sha(path))
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='mismatch'): reader.unchanged()


def test_bad_source_mirror_and_symlink(tmp_path):
    path = tmp_path/'input'
    path.write_text('saved')
    reader = m.EvidenceReader()
    with pytest.raises(ValueError, match='mismatch'): reader.bind(path, '0'*64)
    link = tmp_path/'link'
    link.symlink_to(path)
    with pytest.raises(ValueError, match='nonsymlink'): reader.bind(link)


def test_existing_output_is_never_overwritten(tmp_path):
    output = tmp_path/'exclusive'
    output.mkdir()
    sentinel = output/'report.json'
    sentinel.write_text('user-owned')
    with pytest.raises(FileExistsError): m.run(tmp_path, output)
    assert sentinel.read_text() == 'user-owned'


def test_missing_evidence_persisted_as_rejection(tmp_path):
    output = tmp_path/'new'
    result = m.run(tmp_path, output)
    assert result['status'] == 'invalid_evidence'
    assert result['P2_admission'] is False
    assert m.read_json(output/'report.json') == result
    assert (output/'report.md').is_file()


def test_import_has_no_neural_or_physics_runtime():
    code = ('import sys; from experiments.spectral_reachability import '
            'p2_composition_admission_20261027_v1; '
            'assert not set(sys.modules)&{"torch","gymnasium","stable_worldmodel","mujoco"}')
    subprocess.run([sys.executable, '-B', '-c', code], cwd=m.PROJECT, check=True)


@pytest.fixture
def saved():
    # Only the explicit existing calibration cohort, never future heldout files.
    files = {key:m.PROJECT/'results'/rel for key,rel in m.FILES.items()}
    if not all(p.is_file() for p in files.values()):
        pytest.skip('local historical calibration artifacts unavailable')
    data = {key:m.read_json(p) for key,p in files.items()}
    compilation = m.read_json(m.PROJECT/'results'/m.COMPILATION)
    return data, compilation


def selections(saved, sampling):
    data, compilation = saved
    prefix = 'sampling' if sampling else 'composition'
    return m.validate_planning(data[prefix], data[prefix+'_protocol'], compilation, sampling=sampling)


@pytest.mark.parametrize('sampling', [False, True])
def test_real_saved_root_join_shuffled(saved, sampling):
    data, _ = saved
    selected = selections(saved, sampling)
    physical = deepcopy(data['sampling_physical' if sampling else 'composition_physical'])
    physical['rows'].reverse()
    rows = m.join_physical(selected, physical, sampling=sampling,
                           protocol=data['physical_protocol'] if sampling else None)
    assert len(rows) == 8
    assert not any(r['official_success'] for r in rows)
    assert sum(r['fixed_horizon_complete'] for r in rows) == (8 if sampling else 5)


@pytest.mark.parametrize('defect', ['action', 'goal', 'missing', 'duplicate', 'receipt', 'count', 'state'])
def test_physical_mutations_reject(saved, defect):
    data, _ = saved
    selected = selections(saved, True)
    physical, protocol = deepcopy(data['sampling_physical']), deepcopy(data['physical_protocol'])
    row = physical['rows'][0]
    if defect == 'action': row['trace'][0]['command'][0] += .01
    if defect == 'goal': row['context_identity']['goal_rgb_sha256'] = '0'*64
    if defect == 'missing': physical['rows'].pop()
    if defect == 'duplicate': physical['rows'][-1] = deepcopy(row)
    if defect == 'receipt': protocol['plans'][0]['receipt_sha256'] = '0'*64
    if defect == 'count': row['completed_commands'] -= 1
    if defect == 'state': row['final_state'][0] += 1
    with pytest.raises(ValueError): m.join_physical(selected, physical, sampling=True, protocol=protocol)


@pytest.mark.parametrize('defect', ['fixture', 'entry', 'missing_candidate', 'receipt', 'context', 'protocol'])
def test_saved_sampling_mutations_reject(saved, defect):
    data, compilation = saved
    report = deepcopy(data['sampling'])
    arm = report['rows'][0]['rows'][0]
    if defect == 'fixture': arm['fixture_only'] = True
    if defect == 'entry': arm['actual_model_entries'].pop()
    if defect == 'missing_candidate': arm['attempts'].pop()
    if defect == 'receipt': arm['witness']['receipt']['actions'][0][0] += .01
    if defect == 'context': report['rows'][0]['root']['context']['latent_history'][0][0] += 1
    if defect == 'protocol': report['protocol_sha256'] = '0'*64
    with pytest.raises(ValueError): m.validate_planning(report, data['sampling_protocol'], compilation, sampling=True)


def test_all_existing_evidence_read_only(saved):
    reader = m.EvidenceReader()
    result = m.audit(m.PROJECT, reader)
    reader.unchanged()
    assert result['evidence_integrity_valid']
    assert result['P2_admission'] is False
    assert result['fixed_horizon_branches'] == 13
    assert result['fixed_horizon_composition_flat_pairs'] == 2
    assert len(result['descriptive_comparisons']) == 12
    assert not any(r['full_budget_causal_comparison_admissible'] for r in result['descriptive_comparisons'])
    assert len(reader.hashes) > 100
