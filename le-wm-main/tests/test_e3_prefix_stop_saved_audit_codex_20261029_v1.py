import copy

import pytest

from experiments.spectral_reachability import e3_prefix_stop_saved_audit_codex_20261029_v1 as audit


def event(index, resource, detail, origin='physical'):
    return dict(index=index, resource=resource, detail=detail, origin=origin)


def fixture(n=2, success=False, truncated=True):
    episode = 'cell04'
    events, actions = [], []
    for i in range(1, n+1):
        raw = [0.125, -0.0]
        actions.append(raw)
        bits = audit.raw_bits(raw)
        events.append(event(3*i-2, 'native_authorized_dispatch', dict(episode_id=episode, step=i, raw_bits=bits, command=raw)))
        obs = dict(phase='plan', step=i, observed_event_index=3*i-1, state=[float(i)])
        events.append(event(3*i-1, 'native_fresh_observation', dict(episode_id=episode, receipt=obs)))
        events.append(event(3*i, 'native_control_return', dict(episode_id=episode, step=i, raw_bits=bits, command=raw,
            terminated=i==n and not truncated, truncated=i==n and truncated)))
    row = dict(episode_id=episode, status='native_stopped_prefix', warmup_returned=10,
               control_returned=n, terminated=not truncated, truncated=truncated,
               executed_raw=actions, last_observation=obs, final_state=obs['state'],
               native_success=success, final_metrics=dict(native_success=success), checkpoints=[])
    return events, row, episode


def test_truncated_not_implied_success():
    events, row, episode = fixture()
    result = audit.check(events, row, episode)
    assert result['returned_prefix'] == 2 and result['truncated']
    assert result['recorded_native_success'] is False and not result['native_success_independently_recomputed']
    assert result['feedback_boundaries_completed'] == 0 and not result['full_saved_strict']


@pytest.mark.parametrize('corruption', ['count', 'raw', 'signed_zero', 'flags', 'last_obs', 'state', 'success', 'future', 'future_obs', 'feedback', 'dispatch'])
def test_reject_bad_terminal_evidence(corruption):
    events, row, episode = fixture()
    row = copy.deepcopy(row)
    if corruption == 'count': row['control_returned'] = 1
    if corruption == 'raw': row['executed_raw'][0][0] = 0.25
    if corruption == 'signed_zero': row['executed_raw'][0][1] = 0.0
    if corruption == 'flags': row['truncated'] = False
    if corruption == 'last_obs': row['last_observation']['step'] = 10
    if corruption == 'state': row['final_state'] = [99.0]
    if corruption == 'success': row['native_success'] = True
    if corruption == 'future': row['checkpoints'] = [dict(phase='plan', step=10)]
    if corruption == 'future_obs': events.append(event(10, 'native_fresh_observation', dict(episode_id=episode, receipt=dict(phase='plan', step=10))))
    if corruption == 'feedback': events.append(event(10, 'native_e3_encode_intent', {}))
    if corruption == 'dispatch': events.append(event(10, 'native_authorized_dispatch', {}))
    with pytest.raises((ValueError, KeyError)):
        audit.check(events, row, episode)
