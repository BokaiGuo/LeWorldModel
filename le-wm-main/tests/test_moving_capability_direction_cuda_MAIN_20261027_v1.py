"""RAM contracts, not neural/physical evidence."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import moving_capability_direction_cuda_MAIN_20261027_v1 as m


def fixture():
    h = np.arange(576, dtype=np.float32).reshape(3, 192)
    task = dict(action_mean=[-.0087, .0068], action_std=[.2019, .2002],
                root_context=dict(latent_history=h.tolist(), action_history=np.zeros((2, 10), np.float32).tolist()))
    commands = np.tile(np.asarray([.1, -.1], np.float32), (10, 1))
    prefix = np.ones(192, np.float32)
    open_context = dict(latent_history=np.concatenate([h[1:], np.zeros((1, 192), np.float32)]).tolist(),
                        action_history=[deepcopy(task['root_context']['action_history'][-1]), m.tokens(commands[:5], task)[0].tolist()])
    return task, commands, open_context, prefix


def test_teacher_only_replaces_last_state_and_exact_actions():
    args = fixture(); before = deepcopy(args)
    out = m.contexts(*args)
    np.testing.assert_array_equal(out['teacher'][:2], out['open'][:2])
    np.testing.assert_array_equal(out['teacher'][-1], args[-1])
    np.testing.assert_array_equal(out['open'][-1], np.zeros(192))
    np.testing.assert_array_equal(out['past'][-1], m.tokens(args[1][:5], args[0])[0])
    assert args[0] == before[0] and args[2] == before[2]


@pytest.mark.parametrize('fault', ['history', 'actions', 'dimension'])
def test_context_join_rejects_drift(fault):
    t, a, c, p = fixture()
    if fault == 'history': c['latent_history'][0][0] += 1
    if fault == 'actions': c['action_history'][0][0] += 1
    if fault == 'dimension': p = np.zeros(191)
    with pytest.raises(ValueError): m.contexts(t, a, c, p)


@pytest.mark.parametrize('raw', [np.zeros((4, 2)), np.zeros((5, 3)), np.full((5, 2), 1.1), np.full((5, 2), np.nan)])
def test_bad_raw_packets_refused(raw):
    with pytest.raises(ValueError): m.tokens(raw, fixture()[0])


def test_exact_public_normalization_no_scale100():
    task = fixture()[0]; a = np.full((5, 2), np.float32(.1), dtype=np.float32)
    out = m.tokens(a, task)
    expected = ((a-np.asarray(task['action_mean'], np.float32))/np.asarray(task['action_std'], np.float32)).reshape(1, 10)
    np.testing.assert_array_equal(out, expected)
    assert m.contract()['caps'] == dict(encoder=40, projector=40, action_encoder=80, predictor=80, pred_proj=80)


def test_prepare_external_sha_and_source_drift(tmp_path):
    source = tmp_path/'source'; source.write_text('frozen')
    p = dict(schema=m.NAME+'.prepare', contract=m.contract(), rows=[dict(index=i) for i in range(40)],
             bindings={str(source): m.sha(source)})
    f = tmp_path/'prepare.json'; f.write_bytes(m.rgb.canonical(p))
    expected = m.sha(f)
    assert len(m.verify(f, expected)['rows']) == 40
    with pytest.raises(ValueError): m.verify(f, '0'*64)
    source.write_text('drift')
    with pytest.raises(ValueError): m.verify(f, expected)
