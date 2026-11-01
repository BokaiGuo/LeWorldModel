import copy
import json

import pytest

from experiments.spectral_reachability import e3_cell03_replay_join_codex_20261030_v1 as m


@pytest.fixture
def actual():
    original = [json.loads(x) for x in (m.source.OUTPUT/'events.jsonl').read_bytes().splitlines()]
    replay = [json.loads(x) for x in (m.PHYS/'events.jsonl').read_bytes().splitlines()]
    cuda = json.loads((m.CUDA/'report.json').read_bytes())['rows'][0]
    record = next(x['detail'] for x in original if x['resource'] == 'encoding_complete'
                  and x['detail']['boundary'] == 10)
    row = dict(boundary=10, frame=record['frame'], arrays=record['arrays'])
    return original, replay, row, cuda


def check(actual):
    a, b, c, d = actual
    return m.join_boundary(10, a, b, c, d, m.PHYS, m.CUDA)


def test_real_saved_join(actual):
    assert check(actual)['encoder_recompute_bitwise']


@pytest.mark.parametrize('mode', ['rgb', 'action', 'chronology', 'array', 'missing'])
def test_tamper_fails(actual, mode):
    a, b, c, d = copy.deepcopy(actual)
    if mode == 'rgb':
        c['frame']['pixel_sha256'] = 'wrong'
    elif mode == 'action':
        next(x for x in b if x['kind'] == 'control_return' and x['detail']['step'] == 11)['detail']['raw_bits'] = '0000000000000000'
    elif mode == 'chronology':
        next(x for x in a if x['resource'] == 'native_e3_node_return'
             and x['detail']['boundary'] == 10)['index'] = 1
    elif mode == 'array':
        d['actual_sha256']['latent'] = '0'*64
    else:
        b = [x for x in b if not (x['kind'] == 'control_return' and x['detail']['step'] == 11)]
    with pytest.raises(ValueError):
        check((a, b, c, d))
