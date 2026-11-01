"""Bounded fresh primary-source retrieval and reported-only table export v3."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from lxml import html as lhtml
from experiments.spectral_reachability.d2p_reported_baselines import validate_record, PROTOCOL_FIELDS

NAME = 'p2_reported_baseline_table_20261027_v3'
TEXT_LIMIT = 2 * 1024 * 1024
TOTAL_LIMIT = 24 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[2]
SPECS = {
    'lewm_html': 'https://arxiv.org/html/2603.19312v1',
    'lewm_abs': 'https://arxiv.org/abs/2603.19312v1',
    'hilewm_html': 'https://arxiv.org/html/2607.12547v2',
    'hilewm_abs': 'https://arxiv.org/abs/2607.12547v2',
    'grasp_html': 'https://arxiv.org/html/2602.00475v1',
    'grasp_abs': 'https://arxiv.org/abs/2602.00475v1',
    'planwam_html': 'https://arxiv.org/html/2610.11382v1',
}

# Pins of freshly retrieved first-party bodies, not legacy registry values.
PINS = {
    'grasp_html.html': '2a17a21d05d1792b0fa7b7bd9442be973af6ab37236263994a59a4eb4e6fa899',
    'hilewm_html.html': 'c4c7e725d862017334cbfd72c8e3e6dcdadb1ebff4aae45b3af847ba1e76bc7b',
    'lewm_v3_html.html': '49feca90079f39674b3fe7f23de7cb6e76562c4d4ff5efd0a96743ca668e70b3',
    'lewm_html.html': '1d600ae7a572e90be32a27a4efdfc302bbf7f6b41346f23b2b0068e51c88b92d',
    'planwam_html.html': '67be22fee283aa4bad78578c91d83e6a3085bc8927c63597ee7c9721073bd873',
    'grasp_pdf.pdf': '633bff79e5913401f578d4f13cbee65526221fef859d865cdaab6cfc0fa802a8',
    'hilewm_pdf.pdf': '03f0cf2e39166157c21baebb7808992f3481f9edf2b6cb4d771103e87b586533',
    'lewm_v3_pdf.pdf': '553a01501ac74ec56721cfdb936d3825ac941734133a1fd761f614c01b6887e9',
    'grasp_pdf.txt': '4b2f573b914ca2f40e8826ba18d6ddca8e45ab9b3cb329692d74faacac5c912f',
    'hilewm_pdf.txt': '602131e9ec0f42b51276d3e38a758ba229acf61fa70d49fbc7feda2ec5224faa',
    'lewm_v3_pdf.txt': 'a1f6dc24bf35704e18c53a63a30c8c9b2b3620ed48215d1d9095eebb0054e750',
}
PAPERS = {
    'grasp': ('GRASP', 'Parallel Stochastic Gradient-Based Planning for World Models', '2602.00475v1'),
    'hilewm': ('Hi-LeWM', 'Mind the Gap: Promises and Pitfalls of Hierarchical Planning in LeWorldModel', '2607.12547v2'),
    'lewm_v3': ('LeWM', 'LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels', '2603.19312v3'),
    'planwam': ('PlanWAM', 'PlanWAM: Planning-Shaped Future Representations for End-to-End Autonomous Driving', '2610.11382v1'),
}
QUOTE_IDS = {
    'grasp': ['S4.SS1.p2.1', 'S4.SS2.p1.1', 'S4.SS2.p2.1'],
    'hilewm': ['S2.SS0.SSS0.Px2.p1.1', 'S3.p1.1', 'A3.SS2.p2.1', 'A3.SS2.p3.1', 'A3.SS2.p4.1', 'A3.SS6.p1.1'],
    'lewm_v3': ['A2.p4.1', 'A4.p1.1', 'A4.SS0.SSS0.Px4.p1.1', 'A6.SS1.p1.1'],
    'planwam': ['S4.SS1.SSS0.Px1.p1.1', 'S4.SS1.SSS0.Px2.p1.1', 'A3.SS4.p1.1', 'A3.SS4.p3.2', 'A3.SS4.p3.3'],
}
GAPS = {
    'GRASP': [
        'Published success is within the learned model, not physical simulator execution; define and execute a separate physical protocol.',
        'Obtain exact DINO-wm weights/revision, normalization, start/goal trial IDs and success thresholds.',
        'Reproduce virtual-state relaxation, dynamics/state noise and gradient-detachment contract; local restartAdam is not GRASP.',
        'Disclose best-per-environment/horizon sweep and fix independent evaluation; hardware and full forward/backward/encoding costs remain unknown.',
        'Table 4 timing denominator is not explicit; Table 1 times exclude failures. Do not treat times as matched total pipeline costs.'],
    'Hi-LeWM': [
        'Requires trained macro-action encoder and high-level predictor; low-level LeWM frozen does not mean training-free.',
        'Obtain exact low/high checkpoint revisions, action normalization and training assets; constrained variant also needs the training-action empirical macro library and its permissions/cost.',
        'Match expert trajectory episode IDs, offsets 25/50/75, 50 episodes/config and three seeds; select hyperparameters before held-out outcomes.',
        'Match staged versus online execution and both levels of CEM; Table 5 sample budgets are not full-pipeline costs.',
        'Table 3 is a reproduction by Hi-LeWM authors, not an independent original-LeWM measurement.'],
    'LeWM': [
        'Use arXiv 2603.19312v3 Table 10; v1 lacks that table. Obtain exact evaluated checkpoints/revisions and trial IDs.',
        'Gradient-solver iterations, learning rates, stopping and compute budgets are unknown for Table 10; do not assign CEM 300x30 to Adam/SGD/RMSProp.',
        'Table 10 does not define the plus/minus statistic; seed count and exact link of general evaluation protocol to each solver cell remain unknown.',
        'General Push-T protocol is H5 action blocks, goal offset25, environment budget50, not local fixed-H3/15-step calibration.',
        'Pin success thresholds, preprocessing, action normalization, encoding/verification and optimizer costs before local reproduction.'],
    'PlanWAM': [
        'Driving metrics NAVSIM PDMS/EPDMS and HUGSIM RC/HDScore are not Push-T success and cannot rank local calibration.',
        'Requires trained prior/planner/scorer and rank32 LoRA, four cameras and history frames, NAVSIM datasets and simulator assets with permissions.',
        'Obtain exact checkpoint revisions, episode IDs/seeds, scorer cost and inference hardware; candidate count64 is not total compute.',
        'Official PDF exceeded the 8MiB acquisition cap before body read: PDF page remains unknown, versioned HTML table IDs verified.',
        'HUGSIM Avg is scenario-weighted, not the mean of the four difficulty columns; preserve reported Avg without recomputation.'],
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(data):
    return (json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def check_url(url):
    parts = urlsplit(url)
    if parts.scheme != 'https' or parts.username or parts.password or parts.port or '..' in parts.path.split('/'):
        raise ValueError('unapproved source URL')
    allowed = (parts.hostname == 'arxiv.org' and re.fullmatch(r'/(?:html|abs|pdf)/\d{4}\.\d{4,5}v\d+', parts.path))
    if not allowed or parts.query or parts.fragment:
        raise ValueError('unapproved source URL; only versioned arXiv text pages')
    return True


def bounded(stream, limit):
    raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('response exceeds byte limit')
    return raw


def clean(node):
    # MathML exposes both visible math and TeX annotations; remove annotations
    # to prevent double numbers. Preserve original HTML as the authority.
    return re.sub(r'\s+', ' ', node.text_content()).strip()


def parse_html(text):
    tree = lhtml.fromstring(text, parser=lhtml.HTMLParser(no_network=True))
    for node in tree.xpath('//annotation | //script | //style'):
        node.drop_tree()
    tables = []
    for tab in tree.xpath('//table'):
        figures = tab.xpath('ancestor::figure[1]')
        if not figures or tab.xpath('ancestor::table'):
            continue
        node = figures[0]
        captions = node.xpath('./figcaption')
        rows = [[clean(cell) for cell in tr.xpath('./th|./td')] for tr in tab.xpath('.//tr')
                if tr.xpath('ancestor::table[1]')[0] is tab]
        tables.append({'id': node.get('id', 'unknown'), 'caption': clean(captions[0]) if captions else 'unknown',
                       'table_id': tab.get('id', 'unknown'), 'rows': rows, 'text': clean(tab)})
    paras = [{'id': n.get('id', 'unknown'), 'text': clean(n)} for n in tree.xpath('//p|//h1|//h2|//h3|//h4')]
    links = sorted(set(tree.xpath('//a/@href')))
    return {'title': clean(tree.xpath('//title')[0]) if tree.xpath('//title') else 'unknown',
            'tables': tables, 'paragraphs': paras, 'links': links}


def read_pinned(destination, filename):
    raw = (destination / 'sources' / filename).read_bytes()
    if sha(raw) != PINS[filename]:
        raise ValueError('source SHA mismatch: ' + filename)
    return raw


def numeric(token):
    if token == '—':
        return 'unknown', 'unknown'
    match = re.fullmatch(r'(\d+(?:\.\d+)?)(?:\s*±\s*(\d+(?:\.\d+)?))?s?', token)
    if not match:
        raise ValueError('unrecognized reported token: ' + repr(token))
    return float(match[1]), float(match[2]) if match[2] else 'unknown'


def one_table(doc, figure, table_id=None):
    tables = [t for t in doc['tables'] if t['id'] == figure and (table_id is None or t['table_id'] == table_id)]
    if len(tables) != 1:
        raise ValueError('ambiguous/missing primary table: ' + figure)
    return tables[0]


def build_report(destination):
    """Offline cell extraction, against pinned raw HTML; no legacy report input."""
    docs = {k: parse_html(read_pinned(destination, k + '_html.html').decode('utf-8')) for k in PAPERS}
    if any(docs[k]['title'] != PAPERS[k][1] for k in PAPERS):
        raise ValueError('primary paper title identity mismatch')
    v1 = parse_html(read_pinned(destination, 'lewm_html.html').decode('utf-8'))
    if any(t['caption'].startswith('Table 10:') for t in v1['tables']):
        raise ValueError('unexpected LeWM v1 version contract')
    pdf_hashes = {k: sha(read_pinned(destination, k + '_pdf.pdf')) for k in ('grasp', 'hilewm', 'lewm_v3')}
    pdf_pages = {k: read_pinned(destination, k + '_pdf.txt').decode('utf-8').split('\f') for k in pdf_hashes}
    quotes = {}
    for key, ids in QUOTE_IDS.items():
        quotes[key] = []
        for ident in ids:
            hits = [q for q in docs[key]['paragraphs'] if q['id'] == ident]
            if len(hits) != 1:
                raise ValueError('missing/ambiguous protocol paragraph: ' + ident)
            quotes[key].append(hits[0])
    rows = []

    def emit(key, table, row_index, col, *, method, env, metric='success_rate',
             unit='percent', horizon='unknown', budget='unknown', page='unknown',
             uncertainty_kind='unknown', timing_denominator='unknown', token=None, **details):
        paper, title, version = PAPERS[key]
        raw_row = table['rows'][row_index]
        if page != 'unknown':
            number = re.search(r'Table (\d+):', table['caption'])[1]
            if not re.search(r'Table\s+' + number + r'\b', pdf_pages[key][page-1]):
                raise ValueError('PDF page/table locator mismatch')
        literal = raw_row[col] if token is None else token
        value, spread = numeric(literal)
        protocol = {k: 'unknown' for k in PROTOCOL_FIELDS}
        protocol.update(metric=metric, unit=unit, direction='lower_is_better' if unit == 'seconds' else 'higher_is_better',
                        env=env, task='goal_conditioned_visual_control' if key != 'planwam' else 'autonomous_driving',
                        horizon=horizon, budget=budget,
                        evaluation_protocol={'paragraphs': QUOTE_IDS[key], 'table_caption': table['caption']},
                        uncertainty={'kind': uncertainty_kind if spread != 'unknown' else 'unknown', 'reported_pm': spread})
        if key == 'hilewm':
            protocol.update(n_seeds=3, dataset_split='expert demonstration sampled episodes', observations='current and future goal images')
        elif key == 'planwam':
            protocol.update(dataset_split='navtest' if env.startswith('NAVSIM') else 'HUGSIM public 436 scenarios',
                            observations='4 historical frames x 4 cameras')
        elif key == 'lewm_v3':
            protocol.update(observations='224x224 frames, action blocks of5', dataset_split='offline trajectory sampled start/goal; exact split unknown')
        record = validate_record(dict(paper=paper, title=title, version=version,
            source_url='https://arxiv.org/html/' + version,
            table_or_page=table['id'] + '/' + table['table_id'] + '; PDF page ' + str(page),
            value=value, evidence_kind='paper_reported', **protocol))
        rows.append(dict(id=f'{key}:{table["table_id"]}:{row_index}:{col}', paper_key=key,
            method=method, model='unknown', outcome_domain='learned_world_model_only' if key == 'grasp' else 'paper_environment_evaluation',
            classification='context_only', full_pipeline_cost='unknown', timing_denominator=timing_denominator,
            source=dict(html_file='sources/' + key + '_html.html', html_sha256=PINS[key + '_html.html'],
                figure_id=table['id'], table_id=table['table_id'], table_caption=table['caption'],
                row_index_0based=row_index, column_index_0based=col, raw_row=raw_row,
                reported_token=raw_row[col], parsed_numeric_token=literal, protocol_quotes=quotes[key],
                pdf_page_1based=page, pdf_sha256=pdf_hashes.get(key, 'unknown'),
                pdf_page_text_sha256=sha(pdf_pages[key][page-1].encode()) if page != 'unknown' else 'unknown',
                pdf_url='https://arxiv.org/pdf/' + version if key in pdf_hashes else 'unknown'),
            record=record, **details))

    # GRASP: retain all four planners, five long and nine short task/horizon cells.
    for fig, page, width in [('S4.T1', 7, 11), ('S4.T3', 8, 10), ('S4.T4', 8, 10)]:
        t = one_table(docs['grasp'], fig)
        if len(t['rows']) != 6 or any(len(r) != width for r in t['rows'][2:]):
            raise ValueError('GRASP table shape changed')
        for i, r in enumerate(t['rows'][2:], 2):
            if fig == 'S4.T1':
                for j, h in enumerate((40, 50, 60, 70, 80)):
                    for col, metric, unit in [(1+2*j, 'success_rate', 'percent'), (2+2*j, 'median_success_time', 'seconds')]:
                        emit('grasp', t, i, col, method=r[0], env='Push-T', metric=metric, unit=unit, horizon=h,
                             budget={'trials_per_setting': 500, 'optimizer': 'best swept setting; exact compute unknown'}, page=page,
                             token=r[col].removesuffix(' /'), timing_denominator='successful_trials_only' if unit == 'seconds' else 'not_applicable',
                             feedback='open_loop', horizon_units='paper H; raw-step versus model-step mapping unknown')
            else:
                for j in range(9):
                    emit('grasp', t, i, j+1, method=r[0], env=('Push-T', 'PointMaze', 'WallSingle')[j//3],
                         metric='success_rate' if fig == 'S4.T3' else 'median_completion_time',
                         unit='percent' if fig == 'S4.T3' else 'seconds', horizon=(10, 20, 30)[j%3], page=page,
                         budget={'trials_per_setting': 500, 'optimizer': 'best swept setting; exact compute unknown'}, feedback='unknown',
                         horizon_units='paper H; raw-step versus model-step mapping unknown')

    # Hi-LeWM: preserve all sweep rows, four method/environment columns, plus flat reproduction.
    bt = one_table(docs['hilewm'], 'A3.T5')
    if [len(r) for r in bt['rows']] != [4, 8, 8, 8, 8]:
        raise ValueError('Hi-LeWM budget layout changed')
    budgets = {}
    for r in bt['rows'][2:]:
        nums = list(map(int, r[1:]))
        budgets[int(r[0].split('=')[1])] = dict(environment_steps=nums[0],
            high_cem=dict(zip(('samples', 'steps', 'top_k'), nums[1:4])),
            low_cem=dict(zip(('samples', 'steps', 'top_k'), nums[4:7])))
    t = one_table(docs['hilewm'], 'A3.T4')
    offset = None
    for i, r in enumerate(t['rows'][1:], 1):
        if len(r) == 1 and re.fullmatch(r'd=\d+', r[0]):
            offset = int(r[0].split('=')[1])
            continue
        if len(r) != 11 or offset not in budgets or int(r[1]) != budgets[offset]['environment_steps']:
            raise ValueError('Hi-LeWM sweep layout/goal/budget mismatch')
        for col in range(7, 11):
            emit('hilewm', t, i, col, method='Hi-LeWM' if col%2 else 'Hi-LeWM-C',
                 env='PushT-v1' if col < 9 else 'OGBCube-v0', budget=budgets[offset], page=13,
                 uncertainty_kind='standard_deviation_across_seeds', goal_offset_environment_steps=offset,
                 requires_additional_training=True, schedule_mode='staged' if r[6] == '✓' else 'online',
                 hierarchical_horizons=dict(zip(('high_h', 'high_replan', 'low_h', 'low_r'), map(int, r[2:6]))),
                 budget_source={'figure_id': bt['id'], 'table_id': bt['table_id'], 'pdf_page_1based': 14,
                                'caption': bt['caption'], 'raw_row': next(x for x in bt['rows'][2:] if x[0] == f'd={offset}')})
    for ident, env in [('A3.T3.3', 'PushT-v1'), ('A3.T3.4', 'OGBCube-v0')]:
        t = one_table(docs['hilewm'], 'A3.T3', ident)
        if len(t['rows']) != 9:
            raise ValueError('flat LeWM layout changed')
        offset = None
        for i, r in enumerate(t['rows'][2:], 2):
            if len(r) == 5:
                offset = int(r[0].split('=')[1])
            elif len(r) != 4 or offset is None:
                raise ValueError('flat row span changed')
            fields = r[-4:]
            emit('hilewm', t, i, len(r)-1, method='LeWM (Hi-LeWM paper reproduction)', env=env,
                 budget={'environment_steps': int(fields[0]), 'cem': {'samples': 300, 'steps': 30}},
                 horizon=int(fields[1]), page=13, uncertainty_kind='standard_deviation_across_seeds',
                 goal_offset_environment_steps=offset, receding_horizon_blocks=int(fields[2]),
                 requires_additional_training=False, action_block_size=5)

    # LeWM: Table 10 is v3. General CEM settings are not gradient solver budgets.
    t = one_table(docs['lewm_v3'], 'A7.T10')
    if len(t['rows']) != 5 or any(len(r) != 3 for r in t['rows']):
        raise ValueError('LeWM Table 10 layout changed')
    for i, r in enumerate(t['rows'][1:], 1):
        for col in (1, 2):
            emit('lewm_v3', t, i, col, method=r[0], env='Push-T', page=26,
                 budget={'optimizer': {'samples': 300, 'max_iterations': 30, 'elites': 30} if r[0] == 'CEM' else 'unknown',
                         'scope': 'general CEM protocol; exact per-Table10 solver run binding unknown'},
                 uncertainty_kind='reported_pm_unspecified',
                 general_protocol={'horizon_action_blocks': 5, 'action_block_size': 5, 'goal_offset_environment_steps': 25,
                                   'environment_budget_steps': 50, 'execution_blocks_before_replan': 5,
                                   'binding_to_each_table10_cell': 'unknown'})
            rows[-1]['model'] = t['rows'][0][col]

    # PlanWAM: target method's primary aggregate metrics only; no cross-domain ranking.
    for fig, ident, metric, env in [('S4.T1', 'S4.T1.6', 'EPDMS', 'NAVSIM-v2'),
                                  ('S4.T2', 'S4.T2.1', 'PDMS', 'NAVSIM-v1'),
                                  ('S4.T3', 'S4.T3.1', 'RC', 'HUGSIM'),
                                  ('S4.T3', 'S4.T3.3', 'HDScore', 'HUGSIM')]:
        t = one_table(docs['planwam'], fig, ident)
        found = [(i, r) for i, r in enumerate(t['rows']) if r[0] == 'PlanWAM (ours)']
        if len(found) != 1:
            raise ValueError('PlanWAM target row changed')
        i, r = found[0]
        for col in (range(1, 6) if env == 'HUGSIM' else [len(r)-1]):
            emit('planwam', t, i, col, method=r[0], env=env, metric=metric, unit='score_0_to_100',
                 budget={'candidates': 64, 'waypoints': 8, 'trajectory_seconds': 4, 'pipeline_cost': 'unknown'},
                 difficulty=t['rows'][0][col] if env == 'HUGSIM' else 'not_applicable')
    counts = {name: sum(r['record']['paper'] == name for r in rows) for name in ('GRASP', 'Hi-LeWM', 'LeWM', 'PlanWAM')}
    if counts != {'GRASP': 112, 'Hi-LeWM': 106, 'LeWM': 8, 'PlanWAM': 12} or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('selected table completeness/unique-cell contract failed')
    return dict(schema=NAME, evidence_kind='fresh_primary_paper_reported_only', local_reproduction=False,
        paired_inference_allowed=False, cross_paper_ranking_allowed=False, local_calibration_included=False,
        counts=counts, rows=rows, reproduction_gaps=GAPS,
        version_correction='LeWM v1 has no Table10; selected Table10 is v3, PDF page26.',
        scope='GRASP main Tables1/3/4 all planners; Hi-LeWM Tables3/4 full sweep and Table5 budgets; LeWM v3 Table10 all solvers/models; PlanWAM Tables1/2 target aggregate and Table3 target all difficulties.',
        sources={k: {'html_title': v['title'], 'version': PAPERS[k][2]} for k, v in docs.items()})


def write_export(report, destination):
    """Exclusive offline products, deliberately separate from acquisition artifacts."""
    destination.mkdir(exist_ok=False)
    (destination / 'report.json').write_bytes(encoded(report))
    (destination / 'comparison_readiness.json').write_bytes(encoded({
        'schema': NAME, 'local_reproduction': False, 'gaps': report['reproduction_gaps'],
        'no_cross_paper_ranking': True, 'no_calibration_scores': True}))
    fields = ['paper', 'version', 'method', 'model', 'env', 'metric', 'value', 'reported_pm', 'unit',
              'horizon', 'goal_offset', 'configuration', 'table', 'pdf_page', 'html_url', 'raw_token', 'html_sha256']
    with (destination / 'reported_only.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in report['rows']:
            v, s = r['record'], r['source']
            writer.writerow(dict(paper=v['paper'], version=v['version'], method=r['method'], model=r['model'],
                env=v['env'], metric=v['metric'], value=v['value'], reported_pm=v['uncertainty']['reported_pm'],
                unit=v['unit'], horizon=v['horizon'], goal_offset=r.get('goal_offset_environment_steps', 'unknown'),
                configuration=json.dumps({k: r[k] for k in ('schedule_mode', 'hierarchical_horizons', 'difficulty') if k in r}, ensure_ascii=False),
                table=s['figure_id'] + '/' + s['table_id'], pdf_page=s['pdf_page_1based'],
                html_url=v['source_url'] + '#' + s['figure_id'], raw_token=s['reported_token'], html_sha256=s['html_sha256']))
    for paper in report['counts']:
        lines = [f'# {paper}: primary-source reported values', '',
                 'Reported-only; not local reproduction, equal-compute comparison, or a cross-paper ranking. Full cell protocols and raw rows are in report.json.', '',
                 '| Method/model | Task | Metric | Horizon / offset / configuration | Reported | Source table / PDF page |',
                 '|---|---|---|---|---|---|']
        for r in report['rows']:
            v, s = r['record'], r['source']
            if v['paper'] != paper:
                continue
            config = f"H={v['horizon']}; d={r.get('goal_offset_environment_steps', 'unknown')}"
            for key in ('schedule_mode', 'hierarchical_horizons', 'difficulty'):
                if key in r:
                    config += '; ' + str(r[key])
            lines.append(f"| {r['method']} / {r['model']} | {v['env']} | {v['metric']} ({v['unit']}) | {config} | {s['reported_token']} | [{s['figure_id']}]({v['source_url']}#{s['figure_id']}) / {s['pdf_page_1based']} |")
        (destination / (paper.lower() + '_reported.md')).write_text('\n'.join(lines) + '\n')
    gaps = ['# Exact local reproduction prerequisites', '', 'No local reproduction or P2/P3 admission is asserted.', '']
    for paper, items in report['reproduction_gaps'].items():
        gaps.extend(['## ' + paper, ''] + ['- ' + item for item in items] + [''])
    (destination / 'comparison_readiness.md').write_text('\n'.join(gaps))


def seal_export(destination, exported):
    # Manifest records all acquisition receipts, preserved derivatives, raw bodies,
    # generated table products, and exact new source/tests/doc. No old artifacts edited.
    files = sorted(p for p in destination.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    files += [Path(__file__), ROOT / 'tests' / ('test_' + NAME + '.py'), ROOT / 'docs' / (NAME + '.md')]
    manifest = {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())} for p in files}
    with (exported / 'source_manifest.json').open('xb') as stream:
        stream.write(encoded(manifest))
    sums = [f'{v["sha256"]}  {k}' for k, v in manifest.items()]
    sums.append(sha((exported / 'source_manifest.json').read_bytes()) + '  ' + str((exported / 'source_manifest.json').relative_to(ROOT)))
    with (exported / 'SHA256SUMS').open('x') as stream:
        stream.write('\n'.join(sums) + '\n')


def fetch_sources(destination, supplement=False, pdf=False):
    if not supplement and not pdf:
        destination.mkdir(exist_ok=False)
    sources = destination / 'sources'
    if not supplement and not pdf:
        sources.mkdir()
    receipts = []
    total = sum(p.stat().st_size for p in sources.iterdir() if p.suffix in ('.html', '.pdf'))
    specs = {'lewm_v3_html': 'https://arxiv.org/html/2603.19312v3',
             'lewm_v3_abs': 'https://arxiv.org/abs/2603.19312v3'} if supplement else SPECS
    if pdf:
        specs = {k.replace('_html', '_pdf'): v.replace('/html/', '/pdf/') for k, v in
                 {'lewm_v3_html': 'https://arxiv.org/html/2603.19312v3',
                  'hilewm_html': SPECS['hilewm_html'], 'grasp_html': SPECS['grasp_html'],
                  'planwam_html': SPECS['planwam_html']}.items()}
    for key, url in specs.items():
        check_url(url)
        receipt = {'id': key, 'requested_url': url, 'utc': datetime.now(timezone.utc).isoformat()}
        try:
            request = Request(url, headers={'User-Agent': 'LeWM-literature-verification/1.0', 'Accept-Encoding': 'identity'})
            with urlopen(request, timeout=30) as response:
                check_url(response.url)
                limit = min(8 * 1024 * 1024 if pdf else TEXT_LIMIT, TOTAL_LIMIT - total)
                if int(response.headers.get('Content-Length', 0)) > limit:
                    raise ValueError('declared response exceeds byte limit; body not read')
                raw = bounded(response, limit)
                receipt.update(final_url=response.url, http_status=response.status,
                               content_type=response.headers.get('Content-Type', 'unknown'))
            total += len(raw)
            suffix = '.pdf' if pdf else '.html'
            if pdf and not raw.startswith(b'%PDF-'):
                raise ValueError('not a PDF response')
            with (sources / (key + suffix)).open('xb') as stream:
                stream.write(raw)
            if not pdf:
                parsed = parse_html(raw.decode('utf-8'))
                with (sources / (key + '.parsed.json')).open('xb') as stream:
                    stream.write(encoded(parsed))
            receipt.update(status='retrieved', bytes=len(raw), sha256=sha(raw),
                           file='sources/' + key + suffix)
        except Exception as exc:
            receipt.update(status='unavailable', error=type(exc).__name__ + ': ' + str(exc))
        receipts.append(receipt)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
    with (destination / ('retrieval_pdf.json' if pdf else 'retrieval_supplement.json' if supplement else 'retrieval.json')).open('xb') as stream:
        stream.write(encoded({'requests': receipts, 'raw_response_bytes': total,
                              'per_text_limit': TEXT_LIMIT, 'total_limit': TOTAL_LIMIT}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true')
    parser.add_argument('--supplement', action='store_true')
    parser.add_argument('--pdf', action='store_true')
    parser.add_argument('--export', action='store_true', help='offline pinned-source export; exclusive export/ directory')
    args = parser.parse_args(argv)
    if args.export:
        if args.fetch or args.supplement or args.pdf:
            parser.error('export and retrieval cannot be combined')
        destination = ROOT / 'results' / NAME
        report = build_report(destination)
        exported = destination / 'export'
        write_export(report, exported)
        seal_export(destination, exported)
        print(json.dumps({'output': str(exported), 'counts': report['counts'],
                          'report_sha256': sha((exported / 'report.json').read_bytes())}))
        return 0
    if not args.fetch:
        parser.error('use --fetch for bounded primary source retrieval')
    fetch_sources(ROOT / 'results' / NAME, args.supplement, args.pdf)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
