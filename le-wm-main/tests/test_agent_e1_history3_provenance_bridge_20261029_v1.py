import numpy as np
import pytest
from experiments.spectral_reachability import agent_e1_history3_provenance_bridge_20261029_v1 as bridge


def test_actual_sources():
    r = bridge.build()
    assert len(r['rows']) == 2
    assert all(x['action_intervals'] == [[0,5],[5,10]] for x in r['rows'])
    assert not r['scientific_admission'] and r['inherited_compute_cost'] is None


def test_action_alignment_rejects_permutations():
    raw = np.arange(20, dtype=np.float32).reshape(10,2)/30
    mean = np.array([.1, -.2], np.float32); std = np.array([.4,.7], np.float32)
    result = (raw.reshape(2,10)-np.tile(mean,5))/np.tile(std,5)
    assert bridge.action_join(raw, result, mean, std).tobytes() == result.tobytes()
    for changed in (raw[::-1], raw[:,::-1], np.roll(raw, 5, axis=0)):
        with pytest.raises(ValueError): bridge.action_join(changed, result, mean, std)


@pytest.mark.parametrize('mutation', ['nan','shape','std','token'])
def test_action_rejects_invalid(mutation):
    raw = np.zeros((10,2), np.float32); mean = np.zeros(2, np.float32); std = np.ones(2, np.float32); token = np.zeros((2,10), np.float32)
    if mutation == 'nan': raw[0,0] = np.nan
    if mutation == 'shape': raw = raw[:5]
    if mutation == 'std': std[0] = 0
    if mutation == 'token': token[0,0] = 1
    with pytest.raises(ValueError): bridge.action_join(raw, token, mean, std)


def test_pinned_source_rejects_change(tmp_path):
    p = tmp_path / 'x'; p.write_bytes(b'x')
    with pytest.raises(ValueError): bridge.read(p, '0'*64)
