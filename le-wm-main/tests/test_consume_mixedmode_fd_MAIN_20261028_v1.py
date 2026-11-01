from copy import deepcopy
import json
from pathlib import Path
import pytest
from experiments.spectral_reachability import consume_mixedmode_fd_MAIN_20261028_v1 as m


@pytest.fixture(scope='module')
def saved():
    folder = m.producer.source.ROOT/'results/audit_mixedmode_fd_MAIN_20261028_v1_cuda_MAIN01'
    binding = json.loads((folder/'binding.json').read_bytes())
    return (json.loads(Path(binding['prepared']).read_bytes()),
            json.loads((folder/'report.json').read_bytes()),
            [json.loads(line) for line in (folder/'events.jsonl').read_bytes().splitlines()])


def test_actual_complete_saved_eight_points(saved):
    p,r,e = saved
    assert len(m.analyze(p,r,e)) == 8


@pytest.mark.parametrize('damage', ['missing_coordinate','wrong_derivative','wrong_raw','missing_attempt','unfrozen'])
def test_false_or_partial_evidence_rejected(saved, damage):
    p,r,e = deepcopy(saved)
    if damage == 'missing_coordinate':
        r['comparisons'].pop()
    elif damage == 'wrong_derivative':
        r['comparisons'][0]['fd'][0] += .01
    elif damage == 'wrong_raw':
        r['evaluations'][1]['commands'][0][0] += .1
    elif damage == 'missing_attempt':
        e.pop(0)
    else:
        r['model_after'] = 'changed'
    with pytest.raises(ValueError):
        m.analyze(p,r,e)
