"""Independent saved-array arithmetic audit; no neural or physical execution."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/p1_matched_sampling_cuda_20261027_v1_cuda_main01'
PREP = ROOT / 'results/p1_matched_sampling_cuda_20261027_v1_prepared01'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('index', range(4))
def test_actual_sampling_full_journal(index):
    report = read(RUN / 'report.json')
    assert report['status'] == 'cuda_completed_not_admitted'
    assert report['errors'] == [] and report['model_unchanged']
    assert report['resources']['predict'] == {'attempted': 504, 'completed': 504}
    assert report['resources']['action_encoder'] == {'attempted': 504, 'completed': 504}
    assert not report['P1_admission']
    arm = report['arms'][index]
    path = RUN / arm['report_file']
    assert sha(path) == arm['report_sha256']
    result = read(path)
    path = RUN / arm['journal_file']
    assert sha(path) == arm['journal_sha256']
    events = [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]
    candidates = [e['candidate'] for e in events if e['kind'] == 'candidate_complete']
    assert len(candidates) == 60
    assert [c['index'] for c in candidates] == list(range(60))
    inputs = read(PREP / 'protocol.json')['inputs']
    item = next(t for t in inputs['tasks'] if t['root_sha256'] == arm['root_sha256'])
    task = item['task']
    goal, q = np.asarray(task['goal']), np.asarray(task['projection'])
    for c in candidates:
        raw = np.tanh(np.asarray(c['logits'], dtype=np.float32)).reshape(10, 2)
        np.testing.assert_array_equal(raw, c['commands'])
        assert np.all(np.linalg.norm(np.asarray(c['logits']), axis=-1) <= task['packet_logit_cap'])
        terminal = np.asarray(c['contexts'][-1]['latent_history'][-1])
        assert np.isclose(np.mean((q @ (terminal - goal)) ** 2), c['cost'], rtol=1e-13, atol=1e-14)
        history = task['root_context']
        for k, ctx in enumerate(c['contexts']):
            token = ((raw[5*k:5*k+5] - np.asarray(task['action_mean'], np.float32)) /
                     np.asarray(task['action_std'], np.float32)).reshape(10).tolist()
            assert ctx['latent_history'][:-1] == history['latent_history'][1:]
            assert ctx['action_history'] == history['action_history'][1:] + [token]
            history = ctx
    assert not np.asarray(candidates[0]['logits']).any()
    assert result['winner'] == min(candidates, key=lambda c: (c['cost'], c['index']))
    assert result['witness']['contexts'] == result['winner']['contexts']
    assert result['witness']['commands'] == result['winner']['commands']
    for batch, update in enumerate(e for e in events if e['kind'] == 'elite_update'):
        cs = candidates[batch*10:batch*10+10]
        ids = sorted(range(10), key=lambda i: (cs[i]['cost'], i))[:3]
        elite = np.asarray([cs[i]['logits'] for i in ids])
        assert update['elite_slots'] == ids
        np.testing.assert_array_equal(update['mean'], elite.mean(axis=0))
        np.testing.assert_array_equal(update['sigma'], np.maximum(elite.std(axis=0), .05))
    for phase in ('sampling', 'winner'):
        for resource, recorded in result['resources'][phase].items():
            attempts = [e['event'] for e in events if e['kind'] == 'operation_attempted'
                        and e['event']['phase'] == phase and e['event']['resource'] == resource]
            assert len(attempts) == recorded['attempted'] == recorded['completed']
