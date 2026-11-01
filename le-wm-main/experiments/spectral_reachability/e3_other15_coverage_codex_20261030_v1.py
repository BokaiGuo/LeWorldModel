"""Read-only bounded inventory of E3 known16, excluding cell03 artifacts."""
import hashlib
import json
from pathlib import Path

from experiments.spectral_reachability import agent_e3_transport_publication_20261029_v1 as transport

ROOT = Path('/home/gbk/code/LeWorldModel/le-wm-main')
RESOLVER = 'agent_e3_frozen_model_task_resolver_20261029_v1_prepare_MAIN01.json'
RESOLVER_SHA = 'b44699e6a10bb4c257ed8872dc86b99a37508d7022850f8b4fcd257bcd232b26'
SCHEMA = 'agent_e3_continuous_host_launcher_20261029_v1.authority'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    with path.open('rb') as stream:
        return json.load(stream)


def matrix(path, expected_sha):
    if digest(path) != expected_sha:
        raise ValueError('resolver hash mismatch')
    value = read(path)
    expected = [f'r{r}.g{g}.h{h}.d{d}' for r in range(2) for g in range(2)
                for h in (1, 3) for d in ('Dxy', 'Dx')]
    cells = value['cells']
    if (value['denominator']['cells'] != 16 or len(cells) != 16 or
        [c['index'] for c in cells] != list(range(16)) or
        [c['cell_id'] for c in cells] != expected):
        raise ValueError('fixed denominator identity mismatch')
    return cells


def numerical(results, index):
    receipt = results / f'agent_e3_gd_host_cuda_20261029_v1_consume_cell{index:02d}_MAIN01.json'
    folder = results / f'agent_e3_gd_host_cuda_20261029_v1_cell{index:02d}_MAIN01'
    try:
        r = read(receipt)
        if (r['cell'] != index or r['status'] != 'control5_cell_native_GD_verified_NOT_admitted'
            or r['scientific_admission'] is not False or r['closed_loop_complete'] is not False
            or digest(folder/'report.json') != r['report_sha256']
            or digest(folder/'completion.json') != r['completion_sha256']):
            raise ValueError('receipt identity or join mismatch')
        return {'state': 'saved_receipt_joined', 'receipt': str(receipt),
                'receipt_sha256_observed': digest(receipt), 'full_strict_rebuilt': False}
    except (OSError, KeyError, ValueError, TypeError) as exc:
        return {'state': 'missing_or_invalid', 'detail': str(exc)}


def authorities(results):
    found = {i: [] for i in range(16) if i != 3}
    patterns = ('agent_e3_cell*_feedback_prepare_*_MAIN01.json',
                'e3_cell02_feedback_prepare_*_authority_LOCAL01.json',
                'agent_e3_chunked_host_authority_*_MAIN01.json')
    for path in sorted({p for pattern in patterns for p in results.glob(pattern)}):
        if 'cell03' in path.name.lower():
            continue
        try:
            item = read(path)
            cell = item.get('cell')
            if item.get('schema') != SCHEMA or type(cell) is not int or cell not in found:
                continue
            refs = item['control5']
            joined = (item['resolver']['sha256'] == RESOLVER_SHA and
                      all(digest(Path(refs[k]['path'])) == refs[k]['sha256']
                          for k in ('prepare', 'report', 'completion', 'receipt')))
            found[cell].append({'path': str(path), 'sha256_observed': digest(path),
                                'referenced_bytes_joined': joined,
                                'source_guard_rebuilt': False})
        except (OSError, ValueError, TypeError, KeyError) as exc:
            # Unassignable inputs are not silently treated as a valid authority.
            found.setdefault('unassigned', []).append({'path': str(path), 'error': str(exc)})
    return found


def job(path):
    """Inspect saved transport, with observed hashes, not external admission pins."""
    report_sha = digest(path / 'report.json')
    completion_sha = digest(path / 'completion.json')
    checked = transport.consume(str(path.resolve()), report_sha, completion_sha)
    report = read(path/'report.json')
    events = [json.loads(line) for line in (path/'events.jsonl').read_bytes().splitlines()]
    bindings = [e['detail'] for e in events if e['resource'] == 'launcher_authority_verified']
    returns = [e['detail'] for e in events if e['resource'] == 'launcher_guarded_return']
    cells = {b.get('cell') for b in bindings}
    if len(bindings) != 1 or len(cells) != 1 or type(bindings[0].get('cell')) is not int:
        raise ValueError('ambiguous cell identity')
    cell = bindings[0]['cell']
    if cell not in range(16):
        raise ValueError('cell outside fixed denominator')
    if len(returns) > 1 or (returns and returns[0].get('cell') != cell):
        raise ValueError('terminal cell mismatch')
    nodes = [e['detail'].get('boundary') for e in events if e['resource'] == 'native_e3_node_return']
    if nodes not in ([], [10], [10, 15]):
        raise ValueError('invalid recorded feedback boundary sequence')
    return {'path': str(path), 'cell': cell, 'transport_verified_observed_sha': True,
            'report_sha256_observed': report_sha, 'completion_sha256_observed': completion_sha,
            'publication_status': report['status'], 'publication_complete': checked['complete'],
            'terminal_status_recorded': returns[0].get('status') if returns else None,
            'feedback_node_boundaries_recorded': nodes,
            'full_saved_strict': False, 'scientific_admission': False}


def inventory(results):
    cells = matrix(results/RESOLVER, RESOLVER_SHA)
    auth = authorities(results)
    rows = [{'index': c['index'], 'cell_id': c['cell_id'],
             'resolver_input_status': c['status'],
             'numerical_control5': numerical(results, c['index']),
             'feedback_authorities': auth.get(c['index'], []),
             'online_jobs': []} for c in cells if c['index'] != 3]
    by_index = {r['index']: r for r in rows}
    unassigned = list(auth.get('unassigned', []))
    for path in sorted(results.glob('agent_e3_chunked_identity_host_20261029_v1_*')):
        # The failed cell03 is intentionally not opened by this independent audit.
        if 'cell03' in path.name.lower() or not path.is_dir():
            continue
        try:
            record = job(path)
            if record['cell'] == 3:
                unassigned.append({'path': str(path), 'error': 'excluded cell03 identity'})
            else:
                by_index[record['cell']]['online_jobs'].append(record)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            unassigned.append({'path': str(path), 'error': str(exc)})
    return {'schema': 'e3_other15_coverage_v1', 'fixed_denominator': 16,
            'excluded_cell': 3, 'rows_inspected': 15, 'rows': rows,
            'unassigned_or_invalid': unassigned, 'scope': 'bounded results filename patterns',
            'new_neural_calls': 0, 'new_physics_calls': 0, 'scientific_admission': False}


if __name__ == '__main__':
    print(json.dumps(inventory(ROOT/'results'), sort_keys=True))
