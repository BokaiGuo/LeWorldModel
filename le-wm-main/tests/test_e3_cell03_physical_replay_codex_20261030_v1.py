import json

import pytest

from experiments.spectral_reachability import e3_cell03_physical_replay_codex_20261030_v1 as m


def observation(phase, step):
    return dict(phase=phase, step=step, pixel_sha256='pixels', state=[1],
                snapshot={'state_float64_bits': 'a'}, goal=[2],
                goal_rgb_sha256='goal', goal_pose=[3])


def event(name, detail):
    return dict(origin='physical', resource='native_'+name, detail=detail)


def fixture_events():
    obs = [event('fresh_observation', {'receipt': observation(p, i)})
           for p, n in [('warmup', 11), ('plan', 21)] for i in range(n)]
    warm = [event('warmup_return', dict(step=i)) for i in range(1, 11)]
    controls = [event('control_return', dict(step=i, raw_bits='0000000000000000',
                terminated=False, truncated=False)) for i in range(1, 21)]
    intents = [event('authorized_dispatch', dict(step=i, packet_boundary=((i-1)//5)*5,
               raw_bits='0000000000000000')) for i in range(1, 21)]
    return obs + warm + controls + intents


def test_extract_rejects_missing_reorder_action_and_stop():
    e = fixture_events()
    assert len(m.extract(e)['observations']) == 32
    for mutate in (
        lambda x: x.pop(0),
        lambda x: x[0]['detail']['receipt'].update(step=4),
        lambda x: next(t for t in x if t['resource'] == 'native_authorized_dispatch')['detail'].update(raw_bits='11'),
        lambda x: next(t for t in x if t['resource'] == 'native_control_return')['detail'].update(terminated=True),
    ):
        events = fixture_events()
        mutate(events)
        with pytest.raises(ValueError):
            m.extract(events)


@pytest.mark.parametrize('key', ['pixel_sha256', 'state', 'snapshot', 'goal', 'goal_rgb_sha256', 'goal_pose'])
def test_observation_tamper_fails(key):
    old, new = observation('plan', 1), observation('plan', 1)
    new[key] = None
    with pytest.raises(ValueError, match='observation mismatch'):
        m.matches_observation(old, new)


def test_failure_evidence_precedes_comparison(tmp_path):
    expected = observation('warmup', 0)
    store = m.Store(tmp_path, dict(observations=[expected], warm=[], controls=[]))
    wrong = observation('warmup', 0)
    wrong['pixel_sha256'] = 'other'
    with pytest.raises(ValueError):
        store.emit('fresh_observation', receipt=wrong)
    saved = json.loads((tmp_path/'events.jsonl').read_text())
    assert saved['detail']['receipt']['pixel_sha256'] == 'other'
    assert store.obs == 0


def test_output_no_overwrite_and_cap(tmp_path):
    store = m.Store(tmp_path, dict(observations=[], warm=[], controls=[]))
    store.write('a', b'one')
    with pytest.raises(FileExistsError):
        store.write('a', b'two')
    store.used = m.CAP
    with pytest.raises(ValueError, match='output cap'):
        store.write('other', b'x')


def test_run_retains_failure_and_does_not_overwrite(tmp_path, monkeypatch):
    output = tmp_path/'run'
    monkeypatch.setattr(m, 'OUTPUT', output)
    monkeypatch.setattr(m, 'verify', lambda digest: {'trace': dict(observations=[], warm=[], controls=[])})
    monkeypatch.setattr(m.inputs.host, 'authority', lambda *args: (_ for _ in ()).throw(ValueError('binding drift')))
    assert m.run('fixed') == 1
    report = json.loads((output/'report.json').read_bytes())
    assert report['status'] == 'failed' and report['failure']['message'] == 'binding drift'
    assert [json.loads(x)['kind'] for x in (output/'events.jsonl').read_bytes().splitlines()] == [
        'replay_started', 'replay_failure']
    with pytest.raises(FileExistsError):
        m.run('fixed')
