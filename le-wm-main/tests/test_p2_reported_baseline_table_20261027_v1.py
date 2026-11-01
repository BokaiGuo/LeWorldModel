"""Offline exporter regression; no test values are new literature evidence."""
from pathlib import Path
import json

import pytest

from experiments.spectral_reachability import p2_reported_baseline_table_20261027_v1 as m

ROOT = Path(__file__).resolve().parents[2]


def test_real_verified_docs_project_through_existing_registry():
    report = m.build(ROOT)
    rows = report['reported_only']
    assert len(rows) == 24
    assert {r['record']['paper'] for r in rows} == {'PlanWAM', 'HWM'}
    assert all(r['record']['evidence_kind'] == 'paper_reported' for r in rows)
    assert {r['classification'] for r in rows} == {'context_only', 'insufficient_metadata'}
    assert all(r['same_protocol_with_local'] is False for r in rows)
    assert report['registry_module'].endswith('d2p_reported_baselines.py')
    assert report['local_scores_read'] == 0
    assert report['rank_sota_allowed'] is False
    json.dumps(report, allow_nan=False)


def test_planwam_no_metric_version_or_difficulty_pooling():
    rows = m.build(ROOT)['reported_only']
    p = [r for r in rows if r['record']['paper'] == 'PlanWAM']
    assert [r['record']['value'] for r in p] == [90.9, 93.8, 87.6, 48.3, 51.9, 37.2, 53.7, 75.3, 32.9, 35.6, 22.0, 38.7]
    assert p[0]['record']['env'] == 'NAVSIM-v2'
    assert p[1]['record']['env'] == 'NAVSIM-v1'
    assert p[0]['record']['metric'] == 'EPDMS'
    assert p[1]['record']['metric'] == 'PDMS'
    assert all(r['pdf_page'] == 'unknown' for r in p)
    assert all(r['record']['checkpoint'] == 'unknown' for r in p)
    assert all(r['record']['n_seeds'] == 'unknown' for r in p)
    assert all(r['record']['horizon'] == 'unknown' for r in p)
    assert p[6]['statistical_unit_note'].startswith('scenario-weighted')


def test_hwm_flat_hier_gap_not_rollout_horizon():
    rows = [r for r in m.build(ROOT)['reported_only'] if r['record']['paper'] == 'HWM']
    assert [r['record']['value'] for r in rows] == [84, 55, 17, 89, 78, 61, 100, 84, 44, 100, 95, 83]
    assert [r['goal_distance'] for r in rows[:3]] == ['25', '50', '75']
    assert all(r['record']['horizon'] == 'unknown' for r in rows)
    assert len({r['comparison_group'] for r in rows}) == 6
    assert all(r['record']['version'] == '2604.03208v2' for r in rows)
    assert all(r['record']['evaluation_protocol'] == 'unknown' for r in rows
               if not r['method'].endswith('(hier.)'))


def test_project_values_are_not_paper_registry_records():
    report = m.build(ROOT)
    rows = report['official_project_only']
    assert len(rows) == 22
    assert all(r['evidence_kind'] == 'official_project_reported' for r in rows)
    assert all(r['version'] == 'unknown' for r in rows)
    assert {r['method'] for r in rows if r['system'] == 'AW-0'} == {'Joint denoise', 'Autoregressive code generation', 'Autoregressive'}
    assert all(r['pdf_page'] == 'unknown' for r in rows)
    assert [r['value'] for r in rows] == [71.1, 54.5, 0.23, 0.55, 74.0, 0, 0.7,
        77.3, 12.34, 67.1, 79.3, 0.73, 7.6, 80.0, 3.03, 5.2, 36.0, 30.8, 6.04,
        37.5, 31.9, 5.97]


@pytest.mark.parametrize('method', ['GRASP', 'Hi-LeWM', 'LeWM', 'SAGE', 'SuperMimic', 'PRISM', 'IMWM', 'AP-CEM', 'AP-rank', 'Frozen HWM Subgoal', 'CGS', 'Dual-WM', 'Slot-JEPA', 'HarnessPAI', 'WorldForge'])
def test_unknown_is_not_zero_or_attachment_transcription(method):
    report = m.build(ROOT)
    row = next(r for r in report['reproduction_readiness'] if r['method'] == method)
    assert row['verified_paper_values'] == 0
    assert row['published_value'] == 'unknown'
    assert row['local_reproduction_verified'] is False
    assert row['gaps'] and row['source_anchors']


def test_exact_gap_permissions_and_not_proxy_grasp():
    gaps = {r['method']: r for r in m.build(ROOT)['reproduction_readiness']}
    assert 'restart Adam' in ' '.join(gaps['GRASP']['gaps'])
    assert 'observation memory' in ' '.join(gaps['AP-CEM']['gaps'])
    assert 'recorded action' in ' '.join(gaps['AP-rank']['gaps'])
    assert 'proprioception' in ' '.join(gaps['HWM']['gaps'])
    assert '4-camera' in ' '.join(gaps['PlanWAM']['gaps'])


def test_anchor_ambiguous_missing_and_numeric_corruption_rejected():
    with pytest.raises(ValueError, match='unique'):
        m.one_line('x\nx', 'x')
    with pytest.raises(ValueError, match='unique'):
        m.one_line('y', 'x')
    for token in ['unknown', 'NaN', 'inf', '1 ± 2', 'True', '']:
        with pytest.raises(ValueError):
            m.number(token)


def test_bounded_reader_rejects_escape_symlink_and_oversize(tmp_path):
    with pytest.raises(ValueError):
        m.read_bounded(tmp_path, '../escape')
    (tmp_path / 'big').write_bytes(b'x' * (m.MAX_BYTES + 1))
    with pytest.raises(ValueError):
        m.read_bounded(tmp_path, 'big')
    (tmp_path / 'link').symlink_to('/etc/hosts')
    with pytest.raises(ValueError):
        m.read_bounded(tmp_path, 'link')


def test_exclusive_outputs_and_separate_tables(tmp_path):
    report = m.build(ROOT)
    dest = tmp_path / 'new'
    m.write_outputs(report, dest)
    with pytest.raises(FileExistsError):
        m.write_outputs(report, dest)
    assert len(json.loads((dest / 'reported_only.json').read_text())) == 24
    assert len(json.loads((dest / 'official_project_only.json').read_text())) == 22
    assert 'GRASP' in (dest / 'reproduction_readiness.md').read_text()
    assert '24.938' not in (dest / 'reported_only.csv').read_text()
    assert (dest / 'SHA256SUMS').is_file()


def test_all_export_values_retain_exact_source_cell_and_source_seal():
    report = m.build(ROOT)
    assert report['all_sources_stable'] is True
    for row in report['reported_only']:
        assert float(row['reported_token']) == row['record']['value']
        anchor = row['source_anchor']
        lines = (ROOT / anchor['path']).read_text().splitlines()
        assert lines[anchor['line'] - 1] == anchor['text']
        assert anchor['sha256'] == report['sources'][anchor['path']]['sha256']


def test_known_hwm_budget_scopes_not_replaced_with_unknown_or_multiplied():
    rows = m.build(ROOT)['reported_only']
    flat = next(r for r in rows if r['method'] == 'DINO-WM' and r['goal_distance'] == '75')
    assert flat['record']['budget']['levels'][0]['samples_per_update'] == 1000
    assert flat['record']['budget']['levels'][0]['prediction_horizon'] == 15
    hier = next(r for r in rows if r['method'] == 'DINO-WM (hier.)' and r['goal_distance'] == '75')
    assert len(hier['record']['budget']['levels']) == 2
    assert hier['record']['budget']['levels'][0]['k'] == 'unknown'
    assert hier['record']['budget']['predict_calls_samples'] == 'unknown'
    assert all(r['record']['budget'] == 'unknown' for r in rows
               if r['method'] == 'DINO-WM' and r['goal_distance'] != '75')


def test_verified_document_drift_fails_closed(monkeypatch):
    original = m.read_bounded

    def changed(root, path):
        raw = original(root, path)
        if path.endswith(m.DOCS['plan']):
            return raw.replace(b'90.9', b'91.9')
        return raw

    monkeypatch.setattr(m, 'read_bounded', changed)
    with pytest.raises(ValueError, match='pinned'):
        m.build(ROOT)


def test_exporter_imports_no_execution_libraries():
    import ast
    tree = ast.parse(Path(m.__file__).read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not any(name and name.split('.')[0] in
                   {'torch', 'numpy', 'gym', 'gymnasium', 'requests', 'subprocess', 'urllib'}
                   for name in imports)
