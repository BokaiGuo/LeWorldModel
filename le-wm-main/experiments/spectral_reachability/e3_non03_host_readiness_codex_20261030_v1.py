"""Read-only E3 fixed-denominator readiness and cell02 saved-source checks."""
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import e3_other15_coverage_codex_20261030_v1 as coverage
from experiments.spectral_reachability import e3_cell02_feedback_prepare_codex_20261029_v1 as prep
from experiments.spectral_reachability import agent_e3_control5_authority_join_20261029_v1 as control

ROOT = coverage.ROOT
RESULTS = ROOT / 'results'
PREPARED = RESULTS / 'agent_e3_gd_host_cuda_20261029_v1_prepare_MAIN01.json'
PREPARED_SHA = 'ea8e4feae44e31876852c960db03ce4613184c55fc0745ef8f890f70485f8425'
AUTHORITY = RESULTS / 'e3_cell02_feedback_prepare_codex_20261029_v1_authority_LOCAL01.json'
AUTHORITY_SHA = '596014e1c53b4db19ba3dedc40877d814883e9f07b3b794aec80d7a28d387518'
ROOT_SHA = 'a3bd09470de75652506544b8bea0ff3c5fbae0c0d3b28775da47ee3e1adbb6f7'
CP5_SHA = '5b5f3fb1b83fa832bbbc663d9eb2efcb4171b5d093090a4296e172f3deb2e327'
MISSING = (2, 5, 6, 7, 10, 11, 14, 15)


def control5_saved(index):
    """Join saved inputs with original validator; receipt SHA is observed, not external."""
    folder = RESULTS / f'agent_e3_gd_host_cuda_20261029_v1_cell{index:02d}_MAIN01'
    report = folder/'report.json'
    completion = folder/'completion.json'
    receipt = RESULTS / f'agent_e3_gd_host_cuda_20261029_v1_consume_cell{index:02d}_MAIN01.json'
    return control.load(str(PREPARED), PREPARED_SHA, str(report), coverage.digest(report),
                        str(completion), coverage.digest(completion),
                        str(receipt), coverage.digest(receipt), index)


def cell02():
    """Rebuild fixed authority and join the saved setup, without native execution."""
    result = prep.verify(str(AUTHORITY), AUTHORITY_SHA)
    binding = prep.host.authority(str(AUTHORITY), AUTHORITY_SHA)
    prep.need(binding['cell'] == 2, 'fixed cell02')
    old = prep.host.launch.episode.old
    joined = old.authority.load(*binding['args'])
    body, first = joined['body'], joined['first5']
    prep.need(body['root_sha256'] == ROOT_SHA and body['horizon_packets'] == 3
              and body['domain']['name'] == 'Dxy' and body['feedback_boundary'] == 5,
              'known-root H3 Dxy identity')
    prep.need(first.dtype == np.float32 and first.shape == (5, 2)
              and np.isfinite(first).all() and (np.abs(first) <= 1).all(), 'first5 shape and bounds')
    setup = old.setup_authority.resolve(binding['resolver']['path'], binding['resolver']['sha256'], 2, body)
    cp5 = setup['expected_cp5']
    prep.need(setup['original_root_sha'] == ROOT_SHA and cp5['step'] == 5
              and cp5['pixel_sha256'] == CP5_SHA and setup['prefix5'].shape == (5, 2),
              'saved setup and cp5 identity')
    prep.need(prep.host.launch.host.sha(str(AUTHORITY)) == AUTHORITY_SHA, 'authority unchanged')
    return {'cell': 2, 'task': result['task'], 'authority_rebuilt': True,
            'source_guard_passed': True, 'saved_setup_joined': True,
            'runtime_replay': False, 'scientific_admission': False}


def inspect_all():
    data = coverage.inventory(RESULTS)
    by_index = {r['index']: r for r in data['rows']}
    checks = {}
    for index in MISSING:
        row = by_index[index]
        if any(job['terminal_status_recorded'] is not None for job in row['online_jobs']):
            checks[index] = {'state': 'new_terminal_found_refresh_scope'}
            continue
        try:
            saved = control5_saved(index)
            checks[index] = {'state': 'saved_control5_join_passed',
                             'scientific_admission': saved['scientific_admission'],
                             'feedback_authority_count': len(row['feedback_authorities'])}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            checks[index] = {'state': 'saved_control5_join_failed', 'error': str(exc),
                             'feedback_authority_count': len(row['feedback_authorities'])}
    rows = []
    for index in range(16):
        if index == 3:
            rows.append({'index': 3, 'cell_id': 'r0.g0.h3.dDx',
                         'state': 'excluded_not_inspected'})
        else:
            row = by_index[index]
            rows.append({'index': index, 'cell_id': row['cell_id'],
                         'online_jobs': row['online_jobs'],
                         'missing_online_preconditions': checks.get(index),
                         'feedback_authorities': row['feedback_authorities']})
    return {'fixed_denominator': 16, 'rows': rows,
            'unassigned_or_invalid': data['unassigned_or_invalid'],
            'excluded_cell03_claim_untouched': True, 'production_authorized': False,
            'scientific_admission': False}


def main():
    result = inspect_all()
    try:
        result['cell02_saved_verification'] = cell02()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result['cell02_saved_verification'] = {'state': 'failed', 'error': str(exc)}
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
