"""Offline regressions for bounded primary-source verification."""
import pytest
from experiments.spectral_reachability import p2_reported_baseline_table_20261027_v3 as m


def test_url_boundary_rejects_unapproved_hosts_and_assets():
    for url in ['http://arxiv.org/html/2603.19312v1', 'https://example.com/paper',
                'https://arxiv.org/../../etc/passwd', 'https://huggingface.co/a/weights.pt']:
        with pytest.raises(ValueError):
            m.check_url(url)
    assert m.check_url('https://arxiv.org/html/2603.19312v1')


def test_bounded_stream_does_not_accept_large_body():
    import io
    assert m.bounded(io.BytesIO(b'abc'), 3) == b'abc'
    with pytest.raises(ValueError, match='limit'):
        m.bounded(io.BytesIO(b'abcd'), 3)


def test_tables_preserve_captions_and_column_order():
    html = '<figure id="S1.T1" class="ltx_table"><figcaption>Table 1: test</figcaption><table><tr><th>Method</th><th>Score</th></tr><tr><td>A</td><td>12.3 ± 4.5</td></tr></table></figure>'
    tables = m.parse_html(html)['tables']
    assert tables[0]['id'] == 'S1.T1'
    assert tables[0]['rows'] == [['Method', 'Score'], ['A', '12.3 ± 4.5']]
    assert tables[0]['caption'] == 'Table 1: test'


def test_nested_table_figures_do_not_merge_different_experiments():
    source = '<figure class="ltx_table" id="outer"><figure id="T2"><table class="ltx_tabular" id="T2.t"><tr><td>A</td><td>1</td></tr></table><figcaption>Table 2</figcaption></figure><figure id="T3"><table class="ltx_tabular" id="T3.t"><tr><td>B</td><td>2</td></tr></table><figcaption>Table 3</figcaption></figure></figure>'
    tables = m.parse_html(source)['tables']
    assert [t['id'] for t in tables] == ['T2', 'T3']
    assert [t['rows'] for t in tables] == [[['A', '1']], [['B', '2']]]


@pytest.mark.parametrize('token,value,spread', [('0.0', 0.0, 'unknown'),
    ('96.0±2.83', 96.0, 2.83), ('—', 'unknown', 'unknown'), ('35.3s', 35.3, 'unknown')])
def test_numeric_tokens(token, value, spread):
    assert m.numeric(token) == (value, spread)


@pytest.mark.parametrize('token', ['nan', 'inf', '12 garbage', '', '1/2', '-2'])
def test_no_silent_numeric_coercion(token):
    with pytest.raises(ValueError):
        m.numeric(token)


@pytest.fixture(scope='module')
def report():
    return m.build_report(m.ROOT / 'results' / m.NAME)


def test_complete_selected_tables_not_best_only(report):
    assert report['counts'] == {'GRASP': 112, 'Hi-LeWM': 106, 'LeWM': 8, 'PlanWAM': 12}
    rows = report['rows']
    assert len(rows) == 238
    assert sum(r['record']['value'] == 'unknown' for r in rows) == 2
    assert len({r['id'] for r in rows}) == 238
    assert all(r['record']['evidence_kind'] == 'paper_reported' for r in rows)
    assert report['local_reproduction'] is False


def test_every_cell_has_raw_source_and_protocol(report):
    for row in report['rows']:
        s = row['source']
        assert len(s['html_sha256']) == 64
        assert s['table_id'] and s['raw_row'] and s['reported_token']
        assert s['protocol_quotes']
        assert s['column_index_0based'] > 0
        assert s['pdf_page_1based'] != 'unknown' or row['paper_key'] == 'planwam'
        assert row['record']['checkpoint'] == 'unknown'
        assert row['full_pipeline_cost'] == 'unknown'


def test_lewm_version_and_gradient_budget_not_borrowed(report):
    rows = [r for r in report['rows'] if r['paper_key'] == 'lewm_v3']
    assert {r['record']['version'] for r in rows} == {'2603.19312v3'}
    assert all(r['source']['pdf_page_1based'] == 26 for r in rows)
    adam = next(r for r in rows if r['model'] == 'LeWM' and r['method'].startswith('Adam'))
    assert adam['record']['value'] == 84.0
    assert adam['record']['uncertainty']['kind'] == 'reported_pm_unspecified'
    assert adam['record']['budget']['optimizer'] == 'unknown'
    cem = next(r for r in rows if r['model'] == 'LeWM' and r['method'] == 'CEM')
    assert cem['record']['budget']['optimizer']['samples'] == 300


def test_grasp_is_model_only_and_zero_success_times_remain_missing(report):
    rows = [r for r in report['rows'] if r['paper_key'] == 'grasp']
    assert all(r['outcome_domain'] == 'learned_world_model_only' for r in rows)
    missing = [r for r in rows if r['record']['value'] == 'unknown']
    assert len(missing) == 2
    assert all(r['method'] == 'LatCo' and r['record']['horizon'] in (70, 80) for r in missing)
    assert all(r['timing_denominator'] == 'successful_trials_only' for r in missing)
    assert all(r['timing_denominator'] == 'unknown' for r in rows if r['source']['figure_id'] == 'S4.T4')


def test_hilewm_offset_not_horizon_and_full_staged_online_sweep(report):
    rows = [r for r in report['rows'] if r['paper_key'] == 'hilewm']
    hierarchy = [r for r in rows if r['source']['figure_id'] == 'A3.T4']
    assert len(hierarchy) == 92
    assert {r['goal_offset_environment_steps'] for r in hierarchy} == {25, 50, 75}
    assert {r['schedule_mode'] for r in hierarchy} == {'staged', 'online'}
    assert all(r['record']['horizon'] == 'unknown' for r in hierarchy)
    assert all(r['requires_additional_training'] for r in hierarchy)
    assert sum(r['method'] == 'LeWM (Hi-LeWM paper reproduction)' for r in rows) == 14
    assert {r['record']['budget']['high_cem']['samples'] for r in hierarchy} == {900, 1500, 1200}


def test_planwam_is_separate_domain_with_unknown_pdf_page(report):
    rows = [r for r in report['rows'] if r['paper_key'] == 'planwam']
    assert len(rows) == 12
    assert all(r['source']['pdf_page_1based'] == 'unknown' for r in rows)
    assert {r['record']['metric'] for r in rows} == {'EPDMS', 'PDMS', 'RC', 'HDScore'}
    assert all(r['record']['budget']['candidates'] == 64 for r in rows)


def test_source_corruption_fails_closed(tmp_path):
    (tmp_path / 'sources').mkdir()
    (tmp_path / 'sources' / 'grasp_html.html').write_text('corrupted')
    with pytest.raises(ValueError, match='SHA'):
        m.read_pinned(tmp_path, 'grasp_html.html')


def test_export_is_exclusive_and_does_not_need_network(tmp_path, report, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('network forbidden in export')
    monkeypatch.setattr(m, 'urlopen', forbidden)
    m.write_export(report, tmp_path / 'export')
    assert (tmp_path / 'export' / 'reported_only.csv').exists()
    assert (tmp_path / 'export' / 'comparison_readiness.json').exists()
    with pytest.raises(FileExistsError):
        m.write_export(report, tmp_path / 'export')
