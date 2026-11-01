"""Small pure mock/source tests; fabricated fixture value is not research evidence."""
import copy
import json

import pytest

from experiments.spectral_reachability import d2p_baseline_reported_values_20261027_v1 as m


@pytest.fixture
def row():
    return dict(paper_key='grasp', classification='context_only', full_pipeline_cost='unknown',
        outcome_domain='learned_world_model_only', record=dict(evidence_kind='paper_reported',
        source_url='https://arxiv.org/html/0000.00000v1', version='0000.00000v1', value=1.0),
        source=dict(raw_row=['mock', '1.0'], column_index_0based=1, reported_token='1.0',
        figure_id='mock.T1', table_id='mock.T1.1', protocol_quotes=[dict(text='mock only')],
        html_sha256='0' * 64, pdf_page_1based=1))


def test_mock_only_legal_contract(row):
    assert m.validate_row(row)


@pytest.mark.parametrize('scope,key,value', [
    ('record', 'evidence_kind', 'local_reproduced'),
    ('source', 'reported_token', '2.0'), ('source', 'protocol_quotes', []),
    ('source', 'html_sha256', 'unknown'), ('source', 'pdf_page_1based', 'unknown'),
    ('source', 'pdf_page_1based', 0), ('source', 'pdf_page_1based', True),
    ('record', 'source_url', 'https://example.com/not-primary'),
    ('row', 'classification', 'direct_comparable'), ('row', 'full_pipeline_cost', 0),
    ('row', 'outcome_domain', 'physical_simulator'),
])
def test_mock_rejects_promotion_or_corruption(row, scope, key, value):
    target = row if scope == 'row' else row[scope]
    target[key] = value
    with pytest.raises(ValueError):
        m.validate_row(row)


def test_actual_saved_source_reconstruction():
    result = m.audit()
    assert result['selected_cells'] == 238
    assert result['reported_numeric_cells'] == 236 and result['missing_cells'] == 2
    assert not result['local_reproduction'] and not result['scientific_admission']
    assert result['network_calls'] == result['neural_calls'] == result['physics_calls'] == 0


def test_full_fixture_rejects_forged_pairing():
    saved = json.loads((m.BASE / 'export/report.json').read_text())
    assert m.validate_report(saved)
    for flag in ('local_reproduction', 'local_calibration_included', 'cross_paper_ranking_allowed', 'paired_inference_allowed'):
        changed = copy.deepcopy(saved)
        changed[flag] = True
        with pytest.raises(ValueError):
            m.validate_report(changed)


def test_sage_author_discrepancy_and_weights_unknown_preserved():
    path = m.ROOT.parent / 'docs/p2_sage_primary_source_resolution_20261027_v2.sources.json'
    data = json.loads(path.read_text())
    assert len(data['authenticated_paper']['authors_from_arxiv_citation_metadata']) == 3
    assert len(data['code_release']['readme_authors']) == 4
    assert data['weights_release']['checkpoint_sha256'] is None
    assert data['weights_release']['immutable_revision'] is None
    assert all(r['comparability'] == 'context_only_not_local_paired' for r in data['abstract_reported_values'])


def archive_inputs():
    report = json.loads((m.BASE / 'export/report.json').read_text())
    doc = (m.ROOT.parent / 'docs/d2p_baseline_reported_values_20261027_v1.md').read_text()
    return report['rows'], doc


def test_inline_grasp_values_match_all_original_table1_cells():
    rows, doc = archive_inputs()
    for name in ('CEM', 'LatCo', 'GD', 'GRASP (Ours)'):
        cells = []
        for horizon in (40, 50, 60, 70, 80):
            pair = []
            for metric in ('success_rate', 'median_success_time'):
                row = next(r for r in rows if r['paper_key'] == 'grasp' and r['method'] == name
                           and r['record']['horizon'] == horizon and r['record']['metric'] == metric)
                value = row['record']['value']
                pair.append('unknown' if value == 'unknown' else f'{value:.1f}')
            cells.append(' / '.join(pair))
        label = 'GRASP' if name == 'GRASP (Ours)' else name
        assert '| ' + label + ' | ' + ' | '.join(cells) + ' |' in doc


def test_inline_hilewm_fixed_configuration_not_best_cell_selection():
    rows, doc = archive_inputs()
    config = dict(high_h=2, high_replan=5, low_h=2, low_r=1)
    for mode in ('staged', 'online'):
        for offset in (25, 50, 75):
            selected = [r for r in rows if r['paper_key'] == 'hilewm' and r.get('hierarchical_horizons') == config
                        and r['schedule_mode'] == mode and r['goal_offset_environment_steps'] == offset]
            assert len(selected) == 4
            cells = [f"{r['record']['value']:.1f}±{r['record']['uncertainty']['reported_pm']:.1f}" for r in selected]
            assert f'| {mode} | {offset} | ' + ' | '.join(cells) + ' |' in doc


def test_inline_lewm_and_planwam_values_and_precise_locators():
    rows, doc = archive_inputs()
    for method in ('CEM', 'SGD (β1)', 'RMSProp (β2)', 'Adam (β1,β2)'):
        pair = [r for r in rows if r['paper_key'] == 'lewm_v3' and r['method'] == method]
        assert len(pair) == 2
        assert '| ' + method + ' | ' + ' | '.join(r['source']['reported_token'] for r in pair) + ' |' in doc
    for metric, table in [('RC', 'S4.T3.1'), ('HDScore', 'S4.T3.3')]:
        selected = [r for r in rows if r['paper_key'] == 'planwam' and r['record']['metric'] == metric]
        assert f'| {metric} / {table} | ' + ' | '.join(r['source']['reported_token'] for r in selected) + ' |' in doc
    for row in (r for r in rows if r['paper_key'] == 'planwam'):
        assert row['source']['table_id'] in doc
        assert row['record']['version'] in doc


def test_saved_small_report_matches_actual_offline_check():
    p = m.ROOT.parent / 'docs/d2p_baseline_reported_values_20261027_v1.report.json'
    assert json.loads(p.read_text()) == m.audit()


def test_sage_http_census_and_known_response_cap():
    p = m.ROOT.parent / 'docs/p2_sage_primary_source_resolution_20261027_v2.sources.json'
    d = json.loads(p.read_text())
    receipts, policy = d['new_http_receipts'], d['http_policy']
    assert len(receipts) == policy['request_count'] == 17
    assert sum(r.get('http_status') == 200 for r in receipts) == policy['http_200_count'] == 11
    assert sum(r.get('http_status') == 429 for r in receipts) == policy['http_429_count'] == 2
    assert sum(r['phase'] == 'escalated' for r in receipts) == policy['escalated_requests'] == 14
    assert max(r.get('body_bytes', 0) for r in receipts) == policy['largest_received_body_bytes'] < 65536
