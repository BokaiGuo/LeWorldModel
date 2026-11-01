"""Four-method view of sealed v1 literature evidence; no new value registry.

Missing methods remain explicit summary rows, never fake zero-valued records.
Only the existing v1 paper rows are projected. No local outcomes are read.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path

from experiments.spectral_reachability import d2p_reported_baselines as registry

NAME = 'p2_reported_baseline_table_20261027_v2'
METHODS = ('GRASP', 'Hi-LeWM', 'LeWM', 'PlanWAM')
INPUT = 'results/p2_reported_baseline_table_20261027_v1/report.json'
INPUT_SHA256 = 'f6a0d4f13155a252b85acf413fa7c79d9357b608e14d883426eae2fbe80284c8'
MAX_BYTES = 2_000_000


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def decode_source(raw):
    if len(raw) > MAX_BYTES or sha(raw) != INPUT_SHA256:
        raise ValueError('bounded sealed v1 input SHA mismatch')
    def invalid_constant(value):
        raise ValueError(f'non-finite JSON: {value}')
    return json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=invalid_constant)


def read(root, relative):
    path = root / relative
    if Path(relative).is_absolute() or '..' in Path(relative).parts or path.is_symlink():
        raise ValueError('confined regular source required')
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('source escape')
    with path.open('rb') as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError('source exceeds byte bound')
    return raw


def validate_anchor(anchor, source):
    entry = source['sources'].get(anchor['path'])
    if not entry or anchor['sha256'] != entry['sha256'] or type(anchor['line']) is not int or anchor['line'] < 1:
        raise ValueError('source anchor does not match sealed source manifest')


def project(source):
    """Pure projection; authenticity comes from decode_source's external pin."""
    if source.get('schema') != 'p2_reported_baseline_table_20261027_v1.v1':
        raise ValueError('unexpected upstream schema')
    if source.get('local_scores_read') != 0 or source.get('rank_sota_allowed') is not False:
        raise ValueError('paper-only source required')
    rows, ids = [], set()
    for original in source['reported_only']:
        record = registry.validate_record(original['record'])
        if original['record_id'] in ids:
            raise ValueError('duplicate paper row')
        ids.add(original['record_id'])
        validate_anchor(original['source_anchor'], source)
        if record['paper'] in METHODS:
            if record['paper'] != 'PlanWAM':
                raise ValueError('unexpected new result for previously unverified method')
            if original['classification'] != 'context_only' or original['local_reproduction'] is not False:
                raise ValueError('context-only literature evidence required')
            rows.append(deepcopy(original))
    if len(rows) != 12:
        raise ValueError('complete twelve-row PlanWAM projection required')
    summaries = []
    for method in METHODS:
        found = [r for r in source['reproduction_readiness'] if r['method'] == method]
        if len(found) != 1:
            raise ValueError('unique method prerequisite entry required')
        ready = deepcopy(found[0])
        count = sum(r['record']['paper'] == method for r in rows)
        if ready['verified_paper_values'] != count:
            raise ValueError('source summary count disagreement')
        for anchor in ready['source_anchors']:
            validate_anchor(anchor, source)
        summaries.append({'method': method, 'verified_value_count': count,
                          'value': 'see reported_only' if count else 'unknown',
                          'version': rows[0]['record']['version'] if count else 'unknown',
                          'status': 'reported_context_only' if count else 'missing_verified_result',
                          'local_reproduction': False, 'gaps': ready['gaps'],
                          'source_anchors': ready['source_anchors'],
                          'identity_source_anchors': []})
    return {'schema': NAME + '.v1', 'method_summary': summaries, 'reported_only': rows,
            'protocol_details': deepcopy(source['protocol_details']['plan']),
            'local_scores_read': 0, 'rank_sota_allowed': False,
            'paired_inference_allowed': False, 'total_compute_equivalent': False,
            'P2_admission': False, 'new_primary_verification': False,
            'new_neural_calls': 0, 'new_physical_steps': 0, 'network_used': False,
            'boundary': 'fixed four-method view of existing verified evidence, not new primary verification; no local score table'}


def build(root):
    root = Path(root).resolve()
    raw = read(root, INPUT)
    source = decode_source(raw)
    report = project(source)
    # Verify original source-doc bytes and exact line text, not only embedded
    # anchor hashes. Read only the selected two docs, never a local result.
    selected = {a['path'] for row in report['method_summary'] for a in row['source_anchors']}
    selected |= {row['source_anchor']['path'] for row in report['reported_only']}
    docs = {}
    for relative in sorted(selected):
        doc_raw = read(root.parent, relative)
        if sha(doc_raw) != source['sources'][relative]['sha256']:
            raise ValueError('original verified source doc changed')
        docs[relative] = doc_raw
    for anchor in [r['source_anchor'] for r in report['reported_only']] + [
            a for r in report['method_summary'] for a in r['source_anchors']] + report['protocol_details']:
        validate_anchor(anchor, source)
        if docs[anchor['path']].decode().splitlines()[anchor['line'] - 1] != anchor['text']:
            raise ValueError('source anchor text mismatch')
    for row in report['method_summary']:
        for relative, doc_raw in docs.items():
            # Identity is not result-table verification: LeWM identity can be
            # v1-confirmed while the version of an unverified Table10 is unknown.
            matching = [(i, line) for i, line in enumerate(doc_raw.decode().splitlines(), 1)
                        if line.startswith('| ' + row['method'] + ' |')]
            row['identity_source_anchors'] += [dict(path=relative, line=i, text=line, sha256=sha(doc_raw))
                                               for i, line in matching]
    own_paths = ('experiments/spectral_reachability/' + NAME + '.py',
                 'tests/test_' + NAME + '.py', 'docs/' + NAME + '.md')
    own = {p: read(root, p) for p in own_paths}
    if read(root, INPUT) != raw or any(read(root.parent, p) != b for p, b in docs.items()) or any(
            read(root, p) != b for p, b in own.items()):
        raise ValueError('source changed during export')
    report.update(input_path=INPUT, input_sha256=sha(raw), source_stable=True,
                  own_sources={p: {'sha256': sha(b), 'bytes': len(b)} for p, b in own.items()},
                  original_docs={p: {'sha256': sha(b), 'bytes': len(b)} for p, b in docs.items()})
    return report


def csv_data(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list))
                         else v for k, v in row.items()})
    return stream.getvalue().encode()


def write(report, destination):
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError('symlink destination forbidden')
    destination.mkdir(exist_ok=False)
    flat = [{**r['record'], 'reported_token': r['reported_token'], 'pdf_page': r['pdf_page'],
             'difficulty': r['difficulty'], 'statistical_unit_note': r['statistical_unit_note'],
             'source_anchor': r['source_anchor']} for r in report['reported_only']]
    md = ['# GRASP / Hi-LeWM / LeWM / PlanWAM — reported-only view', '',
          'Missing means unverified, not zero. No local scores or cross-task ranking.', '',
          '| Method | Verified numeric entries | Status | Result version |',
          '| --- | --- | --- | --- |']
    md += [f'| {r["method"]} | {r["verified_value_count"]} | {r["status"]} | {r["version"]} |'
           for r in report['method_summary']]
    md += ['', '## Verified published cells', '',
           '| Method | Version | Table | Environment | Metric | Difficulty | Value | Unit | PDF page |',
           '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    md += ['| ' + ' | '.join(str(r[k]) for k in ('paper', 'version', 'table_or_page', 'env',
           'metric', 'difficulty', 'reported_token', 'unit', 'pdf_page')) + ' |' for r in flat]
    md += ['', '## Exact prerequisites', '']
    md += [f'- {r["method"]}: ' + ' '.join(r['gaps']) for r in report['method_summary']]
    md += ['', 'Full checkpoint / observation / action / budget / seed / source metadata: reported_only.csv and report.json.',
           'Result version unknown for missing values does not deny a separately confirmed paper identity.', '']
    files = {'report.json': encoded(report), 'method_summary.csv': csv_data(report['method_summary']),
             'reported_only.json': encoded(report['reported_only']), 'reported_only.csv': csv_data(flat),
             'table.md': '\n'.join(md).encode()}
    for name, raw in files.items():
        with (destination / name).open('xb') as stream:
            stream.write(raw)
    with (destination / 'SHA256SUMS').open('x') as stream:
        stream.write(''.join(f'{sha(raw)}  {name}\n' for name, raw in sorted(files.items())))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    report = build(root)
    if args.write:
        destination = root / 'results' / NAME
        if not destination.parent.resolve().is_relative_to(root):
            raise ValueError('output escapes project')
        write(report, destination)
        print(destination / 'report.json')
    else:
        print(encoded(report).decode(), end='')
    return 0  # export success, not research admission


if __name__ == '__main__':
    raise SystemExit(main())
