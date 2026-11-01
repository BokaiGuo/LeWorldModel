"""Synthetic pairing and decomposition only, never model evidence."""
from copy import deepcopy
import pytest
from experiments.spectral_reachability.analyze_pusht_observed_prefix_prediction import analyze


def fixture():
    observed = {'status': 'completed', 'gates': {'ok': True}, 'rows': []}
    physical = deepcopy(observed)
    compiled = {'status': 'compiled_calibration', 'gates': {'ok': True}, 'rows': []}
    for i in range(4):
        key = str(i)
        observed['rows'].append({'root_sha256': key, 'root_index': i, 'branches': [
            {'candidate_id': str(j), 'prefix_replay_exact': True, 'observed_prefix_terminal_prediction': [1., 0.]} for j in range(16)]})
        physical['rows'].append({'root_sha256': key, 'branches': [
            {'candidate_id': str(j), 'status': 'executed', 'stages': [{'vector': [1., 0.]}, {'vector': [1., 0.]}]} for j in range(16)]})
        compiled['rows'].append({'root_sha256': key, 'compiled': {'goal_vectors': [[0., 0.]],
            'scoring': {'metric': {'projection': [[1., 0.], [0., 1.]]}},
            'receipts': [{'candidate_id': str(j), 'predicted_terminal_vector': [2., 0.]} for j in range(16)]}})
    return observed, physical, compiled


def test_identity_not_array_order():
    args = fixture()
    result = analyze(*args)
    assert result['improved_candidates'] == 64
    assert result['P1_admission'] is False
    args[1]['rows'].reverse()
    for row in args[1]['rows']: row['branches'].reverse()
    assert analyze(*args) == result


@pytest.mark.parametrize('mutation', ['root', 'duplicate', 'missing', 'replay', 'gate'])
def test_fail_closed(mutation):
    o, p, c = fixture()
    if mutation == 'root': p['rows'][0]['root_sha256'] = 'other'
    if mutation == 'duplicate': o['rows'].append(deepcopy(o['rows'][0]))
    if mutation == 'missing': o['rows'][0]['branches'].pop()
    if mutation == 'replay': o['rows'][0]['branches'][0]['prefix_replay_exact'] = False
    if mutation == 'gate': c['gates']['ok'] = False
    with pytest.raises(ValueError): analyze(o, p, c)
