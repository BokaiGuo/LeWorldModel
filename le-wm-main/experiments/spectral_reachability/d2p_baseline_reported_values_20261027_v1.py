"""Offline archive check, not model evaluation. Existing sealed files stay read-only."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

NAME = 'd2p_baseline_reported_values_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/p2_reported_baseline_table_20261027_v3'
REPORT_SHA = '52e687811e6d6f728ab3eda60836ea05eff11f36da2bb4af975d1dd19ef380f1'
PARSER_SHA = 'b9b975be0f71ec0104a6fd88ca52bdc0b6a22791122e68948280e9b243ff4c40'
COUNTS = {'GRASP': 112, 'Hi-LeWM': 106, 'LeWM': 8, 'PlanWAM': 12}


def require(ok, why):
    if not ok:
        raise ValueError(why)


def validate_row(row):
    """A citeable *reported* cell is never automatically a paired physical result."""
    rec, src = row['record'], row['source']
    require(rec['evidence_kind'] == 'paper_reported', 'reported evidence only')
    require(row['classification'] == 'context_only', 'no unbound comparability promotion')
    require(src['raw_row'][src['column_index_0based']] == src['reported_token'], 'raw cell mismatch')
    require(src['figure_id'] and src['table_id'] and src['protocol_quotes'], 'locator/protocol missing')
    require(len(src['html_sha256']) == 64, 'source SHA missing')
    require(rec['source_url'] == 'https://arxiv.org/html/' + rec['version'], 'versioned primary URL')
    require(row['full_pipeline_cost'] == 'unknown', 'unknown full cost must remain unknown')
    if row['paper_key'] == 'grasp':
        require(row['outcome_domain'] == 'learned_world_model_only', 'model success is not physical success')
    if src['pdf_page_1based'] == 'unknown':
        require(row['paper_key'] == 'planwam', 'missing expected PDF page')
    else:
        require(type(src['pdf_page_1based']) is int and src['pdf_page_1based'] > 0, 'invalid physical page')
    return True


def validate_report(report):
    require(report['counts'] == COUNTS and len(report['rows']) == 238, 'selected coverage changed')
    require(len({r['id'] for r in report['rows']}) == 238, 'duplicate cell identity')
    require(report['local_reproduction'] is False and report['local_calibration_included'] is False
            and report['cross_paper_ranking_allowed'] is False and report['paired_inference_allowed'] is False,
            'reported/local/scientific evidence boundary')
    for row in report['rows']:
        validate_row(row)
    require(sum(r['record']['value'] == 'unknown' for r in report['rows']) == 2, 'missing not zero')
    return True


def audit():
    path = BASE / 'export/report.json'
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == REPORT_SHA, 'sealed export SHA changed')
    parser = ROOT / 'experiments/spectral_reachability/p2_reported_baseline_table_20261027_v3.py'
    require(hashlib.sha256(parser.read_bytes()).hexdigest() == PARSER_SHA, 'upstream parser SHA changed')
    from experiments.spectral_reachability import p2_reported_baseline_table_20261027_v3 as primary
    # Reuse only its offline source extractor. Accidental network access fails.
    with patch.object(primary, 'urlopen', side_effect=AssertionError('network forbidden in offline audit')):
        rebuilt = primary.build_report(BASE)
    saved = json.loads(raw)
    require(rebuilt == saved, 'saved report differs from pinned primary-source reconstruction')
    validate_report(saved)
    saga_path = ROOT.parent / 'docs/p2_sage_primary_source_resolution_20261027_v2.sources.json'
    sage = json.loads(saga_path.read_text())
    require(sage['paper_identity_gap_resolved'] is True and sage['local_reproduction'] is False
            and sage['scientific_admission'] is False, 'SAGE identity is not efficacy')
    require(len(sage['new_http_receipts']) == sage['http_policy']['request_count'], 'receipt census')
    require(all(x.get('body_bytes', 0) <= 65536 for x in sage['new_http_receipts']), 'metadata response cap')
    previous = sage['previous_version']
    for pathkey, shakey in [('doc', 'doc_sha256'), ('source_json', 'source_json_sha256')]:
        require(hashlib.sha256((ROOT.parent / previous[pathkey]).read_bytes()).hexdigest() == previous[shakey],
                'previous SAGE version changed')
    return dict(schema=NAME, status='reported_source_archive_consistent',
                actual_check='offline primary HTML/PDF-text reconstruction equals sealed export',
                methods=COUNTS, selected_cells=238, reported_numeric_cells=236, missing_cells=2,
                mock_is_efficacy=False, local_reproduction=False, scientific_admission=False,
                cross_paper_ranking_allowed=False, report_sha256=REPORT_SHA,
                sage_paper_identity_resolved=True, sage_weight_files_verified=False,
                network_calls=0, neural_calls=0, physics_calls=0)


if __name__ == '__main__':
    print(json.dumps(audit(), ensure_ascii=False, sort_keys=True, indent=2))
