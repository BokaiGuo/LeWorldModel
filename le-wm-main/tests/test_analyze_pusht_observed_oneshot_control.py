"""Synthetic report integrity only; no control efficacy evidence."""
from copy import deepcopy
import pytest
from experiments.spectral_reachability.analyze_pusht_observed_oneshot_control import analyze


def fixture():
    c = {'status': 'completed', 'gates': {'ok': True}, 'rows': []}
    o, t = deepcopy(c), {'status': 'completed', 'rows': []}
    for root in range(4):
        cr = {'root_sha256': str(root), 'root_index': root, 'selected_candidate_id': '0', 'branches': []}
        old = {'root_sha256': str(root), 'selected_candidate_id': '0', 'branches': []}
        task = {'root_sha256': str(root), 'branches': []}
        for index in range(16):
            name = str(index)
            cr['branches'].append({'candidate_id': name, 'prefix_replay_exact': True, 'pose': {'pose_cost': 1.},
                'official': {'official_state_distance': 3., 'official_success': False, 'object_only_success': False}, 'actual_metric_cost': 1.})
            old['branches'].append({'candidate_id': name, 'status': 'executed', 'stages': [{}, {'pose': {'pose_cost': 2.}, 'metric_cost': 2.}]})
            task['branches'].append({'candidate_id': name, 'stages': [{}, {'official_state_distance': 4.}]})
        c['rows'].append(cr); o['rows'].append(old); t['rows'].append(task)
    return c, o, t


def test_all_candidates_and_identity_pairing():
    args = fixture()
    r = analyze(*args)
    assert all(row['pose_better'] == 16 for row in r['rows'])
    assert r['P1_admission'] is False
    args[1]['rows'].reverse()
    for row in args[1]['rows']: row['branches'].reverse()
    assert analyze(*args) == r


@pytest.mark.parametrize('mutation', ['selector', 'root', 'missing', 'replay', 'gate'])
def test_corruption_fails_closed(mutation):
    c, o, t = fixture()
    if mutation == 'selector': c['rows'][0]['selected_candidate_id'] = '1'
    if mutation == 'root': t['rows'][0]['root_sha256'] = 'other'
    if mutation == 'missing': c['rows'][0]['branches'].pop()
    if mutation == 'replay': c['rows'][0]['branches'][0]['prefix_replay_exact'] = False
    if mutation == 'gate': o['gates']['ok'] = False
    with pytest.raises(ValueError): analyze(c, o, t)
