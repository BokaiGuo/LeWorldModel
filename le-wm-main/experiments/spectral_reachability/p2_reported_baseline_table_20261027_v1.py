"""Bounded offline projection of verified-source docs, not another registry.

Only source tables already marked verified are transcribed. Existing
d2p_reported_baselines validates every paper record. No local results are read.
Run from le-wm-main with --write to create a new, exclusive output directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re

from experiments.spectral_reachability import d2p_reported_baselines as registry

NAME = 'p2_reported_baseline_table_20261027_v1'
MAX_BYTES = 2_000_000
UNKNOWN = registry.UNKNOWN
DOCS = {
    'grasp': 'D2P_REPORTED_BASELINES_GRASP_HILEWM_20261026.md',
    'plan': 'D2P_REPORTED_BASELINES_LEWM_PLANWAM_20261026.md',
    'sage': 'D2P_REPORTED_BASELINES_SAGE_SUPERMIMIC_20261026.md',
    'prism': 'D2P_REPORTED_BASELINES_PRISM_IMWM_20261026.md',
    'related': 'D2P_RELATED_SYSTEMS_REPORTED_EVIDENCE_20261026.md',
    'hwm': 'D2P_CROSS_MODEL_BASELINE_ASSET_AUDIT_20261026.md',
    'ap': 'D2P_ANCHORED_PLANNING_PROTOCOL_AUDIT_20261026.md',
    'policy': 'D2P_BASELINE_EVIDENCE_POLICY_20261026.md',
    'faithfulness': 'D2P_BASELINE_FAITHFULNESS_AUDIT_20261026.md',
}
REGISTRY_PATH = 'le-wm-main/experiments/spectral_reachability/d2p_reported_baselines.py'
# Pins freeze already-reviewed source documents, not a second value registry.
VERIFIED_DOC_SHA256 = {
    'grasp': 'f41d703086d74b054b5bb6fa81bf23fff0cc6d2135ae442e5c43dfe2d23216c5',
    'plan': '0612b81763b2190b1f7a48088e6a2403a401afcc40d13eab5931a4e61d05ebaa',
    'sage': 'b33bc9e768c933d44910626d0ea8a9a7d775cf41de23b61b4883847968873d01',
    'prism': 'c8b8598f7c6f33354bbda1dfa2fe4445dd8d48750612c4655a6fecbd4fc380dd',
    'related': 'aaa4b7dcaaf72d79c2b0810ae13db9e1e21db298bd1e8f8c092df83b835b4ad2',
    'hwm': 'e574bb8a4a9a2c8e51183d9d1b70e1a9d6455c35f3bad1af09d69574f4a23f29',
    'ap': '47e7bb5419db6aaee92cf42f32f30b49d948cb5066259bfd6d97da910128e4b3',
    'policy': 'c71afe50b546c2e15f1ed61e772354952ec6bf0b4ca7c4189b590b9b17629edc',
    'faithfulness': '72ceb8015881eeed06df92e74eb42b95cb9b4605c3f5038d4469bbe04cd13702',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_bounded(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if '..' in Path(relative).parts or Path(relative).is_absolute():
        raise ValueError('relative confined source required')
    if not path.resolve().is_relative_to(root) or path.is_symlink():
        raise ValueError('source escapes root or is symlink')
    with path.open('rb') as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError('source exceeds byte bound')
    return raw


def one_line(text, needle):
    matches = [(i, line) for i, line in enumerate(text.splitlines(), 1) if needle in line]
    if len(matches) != 1:
        raise ValueError(f'unique source anchor required: {needle!r}, found {len(matches)}')
    return matches[0]


def number(token):
    if not re.fullmatch(r'\d+(?:\.\d+)?', token):
        raise ValueError(f'plain reported numeric token required: {token!r}')
    return float(token) if '.' in token else int(token)


def cells(line):
    if not line.startswith('|') or not line.endswith('|'):
        raise ValueError('source table row required')
    return [v.strip() for v in line.strip('|').split('|')]


class Sources:
    """Read one fixed small allowlist; do not follow URLs or asset links."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.paths = {k: 'docs/' + v for k, v in DOCS.items()}
        self.paths['registry'] = REGISTRY_PATH
        self.paths['exporter'] = 'le-wm-main/experiments/spectral_reachability/' + NAME + '.py'
        self.paths['tests'] = 'le-wm-main/tests/test_' + NAME + '.py'
        self.paths['doc'] = 'le-wm-main/docs/' + NAME + '.md'
        self.raw = {k: read_bounded(self.root, p) for k, p in self.paths.items()}
        for key, expected in VERIFIED_DOC_SHA256.items():
            if sha(self.raw[key]) != expected:
                raise ValueError(f'pinned verified document changed: {self.paths[key]}')
        self.text = {k: raw.decode('utf-8') for k, raw in self.raw.items()}

    def anchor(self, key, needle):
        line, text = one_line(self.text[key], needle)
        return {'path': self.paths[key], 'line': line, 'text': text,
                'sha256': sha(self.raw[key])}

    def seal(self):
        result = {}
        for key, path in self.paths.items():
            same = read_bounded(self.root, path) == self.raw[key]
            result[path] = {'sha256': sha(self.raw[key]), 'bytes': len(self.raw[key]),
                            'unchanged_on_second_read': same}
            if not same:
                raise ValueError(f'source changed while exporting: {path}')
        return result


def base_record(paper, title, version, url, table, env, task, metric, token):
    result = dict.fromkeys(registry.FIELDS, UNKNOWN)
    result.update(paper=paper, title=title, version=version, source_url=url,
                  table_or_page=table, env=env, task=task, metric=metric,
                  value=number(token), evidence_kind='paper_reported',
                  direction='higher_is_better', unit='0–100 score' if paper == 'PlanWAM' else 'percent')
    return result


def hwm_budget(s, env, method, gap):
    """Only the explicitly reviewed d=75 / hardest-Maze budget cells."""
    hierarchy = method.endswith('(hier.)')
    levels = []
    if env == 'Push-T' and gap == '75':
        for needle in (('| Table 12，hier high |', '| Table 12，hier low |') if hierarchy
                       else ('| Table 11，flat |',)):
            a = s.anchor('hwm', needle)
            c = cells(a['text'])
            levels.append({'source_table_level': c[0], 'elites': number(c[1]),
                           'iterations': number(c[2]), 'samples_per_update': number(c[3]),
                           'prediction_horizon': number(c[4].split('=')[1]),
                           'horizon_symbol': c[4].split('=')[0],
                           'k': number(c[5]) if c[5].isdigit() else UNKNOWN,
                           'source_anchor': a})
    elif env == 'Diverse Maze' and gap == '[13,16]':
        a = s.anchor('hwm', '这些来自原文转存，PDF页码未核实')
        patterns = ((r'Table 14 high为samples(\d+)、pred H(\d+)', 'Table 14 high', 'H'),
                    (r'low为samples(\d+)、pred h(\d+)、k(\d+)', 'Table 14 low', 'h')) if hierarchy else (
                    (r'Table 13 flat MPPI为samples(\d+)、pred h(\d+)、k(\d+)', 'Table 13 flat', 'h'),)
        for pattern, table, symbol in patterns:
            match = re.search(pattern, a['text'])
            if not match:
                raise ValueError('reviewed Maze budget pattern missing')
            values = match.groups()
            levels.append({'source_table_level': table, 'elites': UNKNOWN, 'iterations': UNKNOWN,
                           'samples_per_update': number(values[0]), 'prediction_horizon': number(values[1]),
                           'horizon_symbol': symbol, 'k': number(values[2]) if len(values) == 3 else UNKNOWN,
                           'source_anchor': a})
    if not levels:
        return UNKNOWN
    return {'scope': f'{env}, task distance={gap}, {method} only', 'levels': levels,
            'predict_calls_samples': UNKNOWN, 'encoding_backward_wall_time': UNKNOWN}


def paper_rows(s):
    reg = registry.ReportedBaselineRegistry()
    result = []

    def add(record, method, token, anchor, goal_distance=UNKNOWN, difficulty=UNKNOWN, notes=UNKNOWN):
        rid = f'{record["paper"]}:{record["table_or_page"]}:{method}:{goal_distance}:{difficulty}'
        reg.register(rid, record)
        # Explicit target description; no local scores, checkpoint inference or
        # claim of matching a particular historical calibration run.
        ref = dict.fromkeys(registry.PROTOCOL_FIELDS, UNKNOWN)
        ref['env'] = 'Push-T'
        classification = reg.compare(rid, reference=ref)
        group = {k: record[k] for k in ('paper', 'version', 'env', 'task', 'metric', 'unit', 'dataset_split')}
        group.update(goal_distance=goal_distance, difficulty=difficulty)
        result.append({'record_id': rid, 'record': reg.get(rid), 'method': method,
                       'reported_token': token, 'pdf_page': UNKNOWN,
                       'source_anchor': anchor, 'goal_distance': goal_distance,
                       'difficulty': difficulty, 'comparison_group': sha(json_bytes(group)),
                       'statistical_unit_note': notes, **classification,
                       'reference_protocol': ref,
                       'same_protocol_with_local': False, 'total_compute_equivalent': False})

    title = 'PlanWAM: Planning-Shaped Future Representations for End-to-End Autonomous Driving'
    version = re.search(r'2610\.11382v\d+', s.text['plan']).group()
    url = 'https://arxiv.org/html/' + version
    observations = cells(s.anchor('plan', '| observation |')['text'])[1]
    action = cells(s.anchor('plan', '| action/output |')['text'])[1]
    budget = cells(s.anchor('plan', '| budget |')['text'])[1]
    protocol = cells(s.anchor('plan', '| horizon/反馈 |')['text'])[1]
    for needle, env, metric in (('| Table 1，§4.2，v1 HTML |', 'NAVSIM-v2', 'EPDMS'),
                                ('| Table 2，§4.2，v1 HTML |', 'NAVSIM-v1', 'PDMS')):
        a = s.anchor('plan', needle)
        c = cells(a['text'])
        rec = base_record('PlanWAM', title, version, url, c[0], env, c[1], metric, c[3])
        rec.update(dataset_split='navtest', observations=observations, action_space=action,
                   budget={'reported': budget, 'predict_calls_samples': UNKNOWN,
                           'backward_encoding_wall_time': UNKNOWN}, evaluation_protocol=protocol)
        add(rec, 'PlanWAM', c[3], a, notes=c[4])
    for label, metric, table in (('| RC |', 'Route Completion', 'Table 3(a)'),
                                  ('| HD-Score |', 'HD-Score', 'Table 3(b)')):
        a = s.anchor('plan', label)
        c = cells(a['text'])
        if len(c) != 6:
            raise ValueError('HUGSIM five difficulty/average columns required')
        for difficulty, token in zip(('Easy', 'Medium', 'Hard', 'Extreme', 'Avg.'), c[1:], strict=True):
            rec = base_record('PlanWAM', title, version, url, table, 'HUGSIM',
                              'zero-shot closed-loop driving', metric, token)
            rec.update(dataset_split='436 public scenarios; no HUGSIM-specific fine-tuning',
                       observations=observations, action_space=action,
                       budget={'reported': budget, 'predict_calls_samples': UNKNOWN,
                               'backward_encoding_wall_time': UNKNOWN}, evaluation_protocol=protocol)
            add(rec, 'PlanWAM', token, a, difficulty=difficulty,
                notes='scenario-weighted across 80/157/96/103 scenarios, not training seeds' if difficulty == 'Avg.'
                else 'scenario subset; seed variance and independent training replicates unknown')

    version = re.search(r'2604\.03208v\d+', s.text['hwm']).group()
    for needle, env, methods, gaps in (
        ('| §3.2 Table 2 |', 'Push-T', ('DINO-WM', 'DINO-WM (hier.)'), ('25', '50', '75')),
        ('| §3.3 Table 3 |', 'Diverse Maze', ('PLDM', 'PLDM (hier.)'), ('[5,8]', '[9,12]', '[13,16]')),
    ):
        a = s.anchor('hwm', needle)
        c = cells(a['text'])
        pieces = c[2].split('；')
        if len(pieces) != 2:
            raise ValueError('both flat and hierarchical source methods required')
        for method, piece in zip(methods, pieces, strict=True):
            match = re.fullmatch(re.escape(method) + r' (\d+)/(\d+)/(\d+)%', piece)
            if not match:
                raise ValueError('unexpected HWM verified table format')
            for gap, token in zip(gaps, match.groups(), strict=True):
                rec = base_record('HWM', 'Hierarchical Planning with Latent World Models', version,
                                  'https://arxiv.org/html/' + version, c[0], env, c[1], 'success rate', token)
                rec.update(dataset_split='validation trajectory start–goal pairs' if env == 'Push-T'
                           else '20 held-out maps; 25 training maps',
                           observations=cells(s.anchor('hwm', '| observation/action |')['text'])[1],
                           budget=hwm_budget(s, env, method, gap),
                           evaluation_protocol=cells(s.anchor('hwm', '| horizon/feedback |')['text'])[1]
                           if method.endswith('(hier.)') else UNKNOWN)
                add(rec, method, token, a, goal_distance=gap,
                    notes='task distance is not prediction horizon; seeds, denominators and uncertainty unknown')
    return result


def project_rows(s):
    """Different evidence tier: never accepted by the paper-only registry."""
    result = []

    def add(system, method, metric, token, unit, location, task, anchor, model=UNKNOWN, budget=UNKNOWN):
        result.append(dict(system=system, method=method, model=model, metric=metric,
                           value=number(token), reported_token=token, unit=unit,
                           task=task, env_version=UNKNOWN, checkpoint=UNKNOWN,
                           version=UNKNOWN, pdf_page=UNKNOWN, table_or_page=location,
                           source_url='https://agentic-wam.github.io/' if system == 'AW-0'
                           else 'https://adaptive-visual-tts.github.io/',
                           evidence_kind='official_project_reported', budget=budget,
                           direction='lower_is_better' if metric in ('inference latency', 'NE') else
                           'higher_is_better' if metric in ('rubric score', 'Avg accuracy', 'SR', 'SPL') else
                           'resource_count_not_quality',
                           source_anchor=anchor, classification='context_only',
                           local_reproduction=False, rank_sota_allowed=False))

    specs = (('| Table 3，code generation paradigm |', 'Table 3', 'rubric score', 'percent', '8 real tasks'),
             ('| 同表 | Autoregressive code generation |', 'Table 3', 'rubric score', 'percent', '8 real tasks'),
             ('| Figure 14 |', 'Figure 14', 'inference latency', 'seconds', 'RTX 5090 inference'),
             ('| 同图 | Autoregressive |', 'Figure 14', 'inference latency', 'seconds', 'RTX 5090 inference'))
    for needle, loc, metric, unit, task in specs:
        a = s.anchor('related', needle)
        c = cells(a['text'])
        add('AW-0', c[1], metric, c[3].removesuffix(' s'), unit, loc, task, a)
    for needle in ('| Table 1，SAT-Real |', '| 同表 | GPT-4.1 QA，MindJourney |',
                   '| 同表 | GPT-4.1 QA，AVIC，', '| 同表 | GPT-4.1 QA，AVIC-R，'):
        a = s.anchor('related', needle)
        c = cells(a['text'])
        match = re.fullmatch(r'(?:Avg accuracy )?(\d+\.\d)%，Avg WM (\d+(?:\.\d+)?)，(?:Avg Token )?(\d+\.\d)K(?: tokens)?', c[2])
        if not match:
            raise ValueError('unexpected SAT-Real project table format')
        for metric, token, unit in zip(('Avg accuracy', 'Avg WM', 'Avg Token'), match.groups(),
                                       ('percent', 'project-defined WM unit', 'K tokens'), strict=True):
            add('AVIC', c[1], metric, token, unit, 'Table 1', 'SAT-Real', a,
                model=c[1], budget=c[2])
    for needle in ('| Table 3，R2R |', '| 同表 | MapGPT + AVIC，'):
        a = s.anchor('related', needle)
        c = cells(a['text'])
        match = re.fullmatch(r'SR (\d+\.\d)，SPL (\d+\.\d)，NE (\d+\.\d+)', c[2])
        if not match:
            raise ValueError('unexpected R2R project table format')
        for metric, token in zip(('SR', 'SPL', 'NE'), match.groups(), strict=True):
            add('AVIC', c[1], metric, token, UNKNOWN, 'Table 3', 'R2R', a, model='MapGPT GPT-4o')
    return result


def readiness(s, rows):
    # These are extraction/adapter prerequisites, not claims that an entire
    # machine lacks an asset. Status describes only the named audited sources.
    specs = [
        ('GRASP', 'grasp', '| 完整标题、作者 |', ['Original identity/version/table and official algorithm mapping missing; restart Adam is not official GRASP.']),
        ('Hi-LeWM', 'grasp', '| 完整标题、作者 |', ['Original paper, learned component/checkpoint, task and feedback protocol unverified.']),
        ('LeWM', 'plan', '| Table 10及其他规划表', ['Appendix Table 10 values/protocol unverified; local checkpoint presence does not bind paper checkpoint revision.']),
        ('PlanWAM', 'plan', '| checkpoint/训练 |', ['NAVSIM/HUGSIM evaluator, 4-camera history and trajectory tracking adapter required, not local torque/pusher actions.', 'Frozen DINOv2 base does not freeze trained LoRA/planning modules; published checkpoint revision/license/digest and full inference costs unverified.']),
        ('SAGE', 'sage', '| 完整标题、作者、唯一论文身份 |', ['Primary paper missing; state-assisted Table 1 and visual-only Table 18 remain separate unverified attachment claims.', 'Goal offset is not prediction horizon; manifest SD is not training-seed SEM; module weights/training costs unknown.']),
        ('SuperMimic', 'sage', '| 完整标题、作者、唯一论文身份 |', ['Primary identity/task/control metric/checkpoint and resource protocol unverified.']),
        ('PRISM', 'prism', '| 完整标题/作者/唯一论文身份 |', ['Table 1 values, trained prior assets, MPPI settings and accounting unverified.']),
        ('IMWM', 'prism', '| 完整标题/作者/唯一论文身份 |', ['Table 2 values, intuition weights, retrieval library permissions/leakage and evaluation-cell independence unverified.']),
        ('AP-CEM', 'ap', '| AP 原文 |', ['Primary Table 1 and official code revision missing; observation memory split/index/encoding and query exclusion require binding.', 'Whether recorded actions are read is unverified; do not infer observation-only permissions from CEM name.']),
        ('AP-rank', 'ap', '| AP 原文 |', ['Primary Table 1, recorded action segment access/extraction and standard versus perturbed query manifests require separate binding.']),
        ('HWM', 'hwm', '| HWM | Wancong', ['DINO-WM plus proprioception / PLDM backbones are not local LeWM; published checkpoint revision/license/normalizer and query manifests unverified.', 'Task distance d/D differs from prediction horizon; hierarchical trained modules, layered feedback and full costs need an adapter and separate accounting.']),
        ('Frozen HWM Subgoal', 'hwm', '| Frozen HWM Subgoal | *', ['Full paper Table 2, H-JEPA checkpoint identity, condition-head weights, train split and evaluation statistics unverified.']),
        ('CGS', 'hwm', '| CGS | Ziang', ['Full table/protocol, changed training objective, weights/license and iterations unverified; 128 candidates alone is not a compute budget.']),
        ('Dual-WM', 'hwm', '| Dual-WM | Delin', ['Core code link is advertised only; release/license/task-specific dual-latent checkpoints and full task protocol unverified.']),
        ('Slot-JEPA', 'hwm', '| Slot-JEPA | Peisong', ['Full paper table, slot/mask interface, TMP memory, UOT differentiation, weights/license and task protocol unverified.']),
        ('HarnessPAI', 'related', '| HarnessPAI |', ['Unique paper identity, version, code/weights and control protocol unverified.']),
        ('WorldForge', 'related', '| WorldForge |', ['Unique paper identity, version, code/weights and control protocol unverified; name collision possible.']),
        ('AW-0', 'related', '| AW-0 | 成功', ['Only official-project values verified; paper version/PDF table, robot/memory permissions, execution chunks, checkpoint revision/license unverified.']),
        ('AVIC', 'related', '| AVIC / AVIC-R |', ['Only official-project values verified; AVIC-R has trained policy, unlike AVIC; QA/WM bundle revisions, evaluation manifests and cost boundary unverified.']),
    ]
    output = []
    for method, key, needle, gaps in specs:
        count = sum(r['record']['paper'] == method for r in rows)
        output.append({'method': method, 'verified_paper_values': count,
                       'verified_project_values': 4 if method == 'AW-0' else 18 if method == 'AVIC' else 0,
                       'project_values_ref': 'official_project_only' if method in ('AW-0', 'AVIC') else UNKNOWN,
                       'published_value': 'see reported_only' if count else UNKNOWN,
                       'local_reproduction_verified': False,
                       'readiness': 'not_established_in_inspected_sources', 'gaps': gaps,
                       'source_anchors': [s.anchor(key, needle)],
                       'scope': 'source-doc projection; no repository-wide absence proof or environment reset'})
    return output


def build(root):
    s = Sources(root)
    rows = paper_rows(s)
    project = project_rows(s)
    gaps = readiness(s, rows)
    # Keep exact budget/training/permission text, including its stated unknowns;
    # never derive predict calls from candidate counts or a task gap.
    protocol_details = {}
    for key in ('plan', 'hwm'):
        protocol_details[key] = [s.anchor(key, line) for line in s.text[key].splitlines()
                                 if line.startswith('|') and any(word in line for word in
                                 ('checkpoint', 'observation', 'action/output', 'horizon/反馈',
                                  'budget', 'seed/统计', 'training/split', 'metric/seed',
                                  'Table 11，flat', 'Table 12，hier'))]
    protocol_details['hwm'].append(s.anchor('hwm', '这些来自原文转存，PDF页码未核实'))
    protocol_details['hwm'].append(s.anchor('hwm', '| HWM | Wancong'))
    return {'schema': NAME + '.v1', 'registry_module': REGISTRY_PATH,
            'reported_only': rows, 'official_project_only': project,
            'reproduction_readiness': gaps, 'protocol_details': protocol_details,
            'local_baseline_requirements': [s.anchor('faithfulness', line) for line in
                s.text['faithfulness'].splitlines() if line.startswith('|') and
                any(line.startswith('| ' + label + ' |') for label in
                    ('CEM', 'MPPI', 'vanilla gradient MPC', 'local GN', 'faithful GRASP',
                     'SRA without composition', 'kinodynamic search', 'arbitrary latent waypoint',
                     'learned subgoal prior / SAGE / Hi-LeWM'))],
            'sources': s.seal(), 'all_sources_stable': True,
            'verified_document_pins': VERIFIED_DOC_SHA256,
            'scope': 'bounded existing-source transcription, not new primary verification or local reproduction',
            'primary_source_reverified_this_run': False, 'local_scores_read': 0,
            'rank_sota_allowed': False, 'cross_paper_aggregation_allowed': False,
            'paired_inference_allowed': False, 'total_compute_equivalent': False,
            'P2_admission': False, 'new_neural_calls': 0, 'new_physical_steps': 0,
            'network_used': False}


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()


def flat_paper(row):
    return {**row['record'], **{k: row[k] for k in ('record_id', 'method', 'reported_token',
            'pdf_page', 'goal_distance', 'difficulty', 'comparison_group', 'classification',
            'statistical_unit_note')},
            'source_doc': row['source_anchor']['path'], 'source_line': row['source_anchor']['line'],
            'source_sha256': row['source_anchor']['sha256']}


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list))
                         else v for k, v in row.items()})
    return stream.getvalue().encode()


def markdown(rows, columns, title):
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    lines = ['# ' + title, '', 'Reported individually; no ranking, pooling or local score comparison.', '',
             '| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join('---' for _ in columns) + ' |']
    lines += ['| ' + ' | '.join(cell(row[k]) for k in columns) + ' |' for row in rows]
    return ('\n'.join(lines) + '\n').encode()


def write_outputs(report, destination):
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError('symlink output forbidden')
    destination.mkdir(exist_ok=False)
    paper = [flat_paper(row) for row in report['reported_only']]
    project = report['official_project_only']
    gaps = report['reproduction_readiness']
    files = {
        'report.json': json_bytes(report),
        'reported_only.json': json_bytes(report['reported_only']),
        'reported_only.csv': csv_bytes(paper),
        'reported_only.md': markdown(paper, ['paper', 'version', 'table_or_page', 'method', 'env',
            'metric', 'goal_distance', 'difficulty', 'reported_token', 'unit', 'source_doc', 'source_line'],
            'Verified paper-reported values only'),
        'official_project_only.json': json_bytes(project),
        'official_project_only.csv': csv_bytes(project),
        'official_project_only.md': markdown(project, ['system', 'method', 'task', 'metric',
            'reported_token', 'unit', 'table_or_page', 'source_url'], 'Official project values, NOT verified paper tables'),
        'reproduction_readiness.json': json_bytes(gaps),
        'reproduction_readiness.md': markdown(gaps, ['method', 'verified_paper_values', 'readiness',
            'gaps'], 'Exact local reproduction prerequisites, not evidence of reproduction'),
    }
    for name, raw in files.items():
        with (destination / name).open('xb') as stream:
            stream.write(raw)
    sums = ''.join(f'{sha(raw)}  {name}\n' for name, raw in sorted(files.items()))
    with (destination / 'SHA256SUMS').open('x') as stream:
        stream.write(sums)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args(argv)
    report = build(args.project_root)
    if args.write:
        dest = args.project_root.resolve() / 'le-wm-main/results' / NAME
        if not dest.parent.resolve().is_relative_to(args.project_root.resolve()):
            raise ValueError('output escapes project')
        write_outputs(report, dest)
        print(json.dumps({'output': str(dest), 'paper_values': len(report['reported_only']),
                          'project_values': len(report['official_project_only']), 'P2_admission': False}))
    else:
        print(json_bytes(report).decode(), end='')
    return 0  # export success only, never scientific admission


if __name__ == '__main__':
    raise SystemExit(main())
