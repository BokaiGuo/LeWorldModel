"""Four-method sealed-evidence view; no neural or network execution."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import p2_reported_baseline_table_20261027_v2 as m

ROOT = Path(__file__).resolve().parents[1]


def test_four_methods_visible_without_inventing_three_missing_results():
    report = m.build(ROOT)
    assert [r['method'] for r in report['method_summary']] == list(m.METHODS)
    assert [r['verified_value_count'] for r in report['method_summary']] == [0, 0, 0, 12]
    for row in report['method_summary'][:3]:
        assert row['value'] == 'unknown'
        assert row['status'] == 'missing_verified_result'
        assert row['gaps'] and row['source_anchors']
    assert report['local_scores_read'] == 0
    assert report['rank_sota_allowed'] is False
    assert report['new_primary_verification'] is False


def test_exact_values_and_metadata_are_copied_not_recalculated():
    report = m.build(ROOT)
    upstream = json.loads((ROOT / m.INPUT).read_bytes())
    expected = [r for r in upstream['reported_only'] if r['record']['paper'] == 'PlanWAM']
    assert report['reported_only'] == expected
    assert [r['record']['value'] for r in expected] == [90.9, 93.8, 87.6, 48.3, 51.9, 37.2, 53.7,
                                                       75.3, 32.9, 35.6, 22.0, 38.7]
    assert expected[0]['record']['env'] == 'NAVSIM-v2'
    assert expected[1]['record']['env'] == 'NAVSIM-v1'
    assert all(r['record']['checkpoint'] == 'unknown' and r['pdf_page'] == 'unknown' for r in expected)
    assert report['protocol_details'] == upstream['protocol_details']['plan']


def test_grasp_is_not_restart_adam_and_lewm_identity_not_table_verification():
    summary = {r['method']: r for r in m.build(ROOT)['method_summary']}
    assert 'restart Adam' in ' '.join(summary['GRASP']['gaps'])
    assert summary['LeWM']['version'] == 'unknown'
    assert summary['LeWM']['identity_source_anchors']
    assert 'Table 10' in ' '.join(summary['LeWM']['gaps'])


@pytest.mark.parametrize('mutation', ['local', 'duplicate', 'missing_plan', 'extra_target', 'nonfinite'])
def test_invalid_projection_inputs_fail(mutation):
    source = json.loads((ROOT / m.INPUT).read_bytes())
    if mutation == 'local':
        source['reported_only'][0]['record']['evidence_kind'] = 'local_reproduction'
    elif mutation == 'duplicate':
        source['reported_only'].append(deepcopy(source['reported_only'][0]))
    elif mutation == 'missing_plan':
        source['reported_only'].pop(0)
    elif mutation == 'extra_target':
        row = deepcopy(source['reported_only'][0])
        row['record']['paper'] = 'GRASP'
        source['reported_only'].append(row)
    else:
        source['reported_only'][0]['record']['value'] = float('nan')
    with pytest.raises(ValueError):
        m.project(source)


def test_source_pin_and_bounded_json_reject_drift_and_duplicate_keys():
    raw = (ROOT / m.INPUT).read_bytes()
    assert m.decode_source(raw)['schema'].endswith('.v1')
    for invalid in [raw + b' ', b'{"a":1,"a":2}', b'x' * (m.MAX_BYTES + 1)]:
        with pytest.raises(ValueError):
            m.decode_source(invalid)


def test_source_anchor_hash_mismatch_rejected():
    source = json.loads((ROOT / m.INPUT).read_bytes())
    source['reported_only'][0]['source_anchor']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='anchor'):
        m.project(source)


def test_independent_exclusive_files_and_no_other_methods(tmp_path):
    report = m.build(ROOT)
    dest = tmp_path / 'fresh'
    m.write(report, dest)
    with pytest.raises(FileExistsError):
        m.write(report, dest)
    text = (dest / 'table.md').read_text()
    assert all(name in text for name in m.METHODS)
    assert 'HWM' not in text and 'AVIC' not in text
    assert 'calibration' not in (dest / 'reported_only.csv').read_text()
    assert (dest / 'SHA256SUMS').is_file()


def test_only_new_module_dependencies_sealed():
    result = m.build(ROOT)
    assert result['input_sha256'] == m.INPUT_SHA256
    assert result['source_stable'] is True
    assert len(result['own_sources']) == 3
    assert all(len(item['sha256']) == 64 for item in result['own_sources'].values())
