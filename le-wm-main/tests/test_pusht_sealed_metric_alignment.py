"""Synthetic algebra/identity contracts only; no neural efficacy evidence."""
from copy import deepcopy
import pytest
from experiments.spectral_reachability.analyze_pusht_sealed_metric_alignment import analyze


def fixture():
    receipts = [{'candidate_id': str(i), 'program_live_commands': [[0., 0.], [0., 0.]]} for i in range(16)]
    branches = [{'candidate_id': str(i), 'commands': r['program_live_commands'], 'status': 'executed',
                 'stages': [{'vector': [float(i), 1.], 'metric_cost': i*i/2,
                             'pose': {'pose_cost': float(i)}} for _ in range(2)]} for i, r in enumerate(receipts)]
    compiled = {'goal_vectors': [[0., 0.]], 'scoring': {'metric': {'projection': [[1., 0.], [0., 0.]]}}, 'receipts': receipts}
    e = {'status': 'completed', 'gates': {'ok': True}, 'rows': [{'root_sha256': 'a', 'root_index': 0, 'branches': branches}]}
    c = {'status': 'compiled_calibration', 'gates': {'ok': True}, 'rows': [{'root_sha256': 'a', 'compiled': compiled}]}
    return e, c


def test_energy_and_pairing():
    e, c = fixture()
    before = deepcopy((e, c))
    r = analyze(e, c)
    assert r['rows'][0]['candidates'][3]['stages'][0]['removed'] == .5
    e['rows'][0]['branches'].reverse()
    assert analyze(e, c) == r
    e['rows'][0]['branches'].reverse()
    assert (e, c) == before
    assert r['P1_admission'] is False


@pytest.mark.parametrize('mutation', ['duplicate', 'root', 'cost', 'command', 'failure', 'gate'])
def test_corruptions_fail_closed(mutation):
    e, c = fixture()
    b = e['rows'][0]['branches'][0]
    if mutation == 'duplicate': e['rows'].append(deepcopy(e['rows'][0]))
    if mutation == 'root': e['rows'][0]['root_sha256'] = 'b'
    if mutation == 'cost': b['stages'][0]['metric_cost'] = 50
    if mutation == 'command': b['commands'] = [[.1, 0.], [0., 0.]]
    if mutation == 'failure': b['status'] = 'failed'
    if mutation == 'gate': e['gates']['ok'] = False
    with pytest.raises(ValueError): analyze(e, c)
