from copy import deepcopy
import pytest

from experiments.spectral_reachability import p2_saved_fair_pair_20261029 as m


@pytest.fixture
def inputs():
    return [m.read(m.ROOT/'results'/path, sha) for path, sha in m.SOURCES.values()]


def test_actual_six_slots_and_early_stop(inputs):
    out = m.analyze(*inputs)
    assert (out['total_pairs'], out['comparable']) == (18, 16)
    missing = [x for x in out['pairs'] if not x['comparable']]
    assert len(missing) == 2
    assert all(x['root'] == 1 and x['step'] == 15 and x['native7_right_minus_left'] is None for x in missing)
    assert out['equal_flops'] is None and not out['no_composition_sra']['comparable']


@pytest.mark.parametrize('field', ['goal', 'root_state', 'goal_pixel_sha256'])
def test_mismatch_never_ranked(inputs, field):
    p, r = deepcopy(inputs)
    p['rows'][1][field] = 'different'
    if field == 'root_state':
        p['rows'][1]['checkpoints'][0]['state'] = 'different'
    out = m.analyze(p, r)
    affected = [x for x in out['pairs'] if x['root'] == 0 and 'composition' in (x['left'], x['right'])]
    assert all(not x['comparable'] and x['native7_right_minus_left'] is None for x in affected)


def test_task_and_horizon_cannot_be_spliced(inputs):
    p, r = deepcopy(inputs)
    p['rows'][0]['plan']['raw_steps'] = 10
    with pytest.raises(ValueError, match='horizon'):
        m.analyze(p, r)
    p, r = deepcopy(inputs)
    r['rows'][0]['task_id'] = 'different'
    with pytest.raises(ValueError, match='task binding'):
        m.analyze(p, r)


def test_bad_checkpoint_coverage_rejected(inputs):
    p, r = deepcopy(inputs)
    p['rows'][5]['checkpoints'].append(deepcopy(p['rows'][5]['checkpoints'][-1]))
    with pytest.raises(ValueError, match='checkpoint coverage'):
        m.analyze(p, r)


def test_pinned_hash_detects_tampering(tmp_path):
    path = tmp_path/'data.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='SHA mismatch'):
        m.read(path, '0'*64)
