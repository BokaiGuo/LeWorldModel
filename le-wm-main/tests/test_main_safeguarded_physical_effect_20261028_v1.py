from copy import deepcopy

import pytest

from experiments.spectral_reachability import main_safeguarded_physical_effect_20261028_v1 as m


def fixture():
    row = dict(index=0, root_slot=0, condition='off', arm='test', task_id='task', goal=[0.]*7,
        root_state=[0.]*7, status='horizon_completed', own_returned=10,
        checkpoints=[dict(step=5, snapshot='exact', state=[0.]*7)], final_state=[3., 4., 0., 0., 0., 0., 0.])
    newer = deepcopy(row); newer['final_state'][0] = 0.
    a, b = [m.base.metrics(x['final_state'], x['goal']) for x in (row, newer)]
    pair = dict(index=0, control=dict(final_metrics=a), intervention=dict(final_metrics=b),
        new_minus_control={k:b[k]-a[k] for k in m.base.AXES},
        safeguard=dict(selected_index=2, incumbent_Q=2., selected_Q=1.))
    compile_row = dict(index=0, selected_index=2, evaluations=[dict(cost=x) for x in (2., 3., 1., 4., 5.)])
    return row, newer, pair, compile_row


def test_real_selected_index_not_last_or_first_proposal():
    r = m.compare(*fixture())
    assert r['selected_Q'] == 1. and r['predicted_Q_delta'] == -1.
    assert r['new_minus_control']['joint_xy_error'] == -1.


@pytest.mark.parametrize('change', ['prefix', 'early', 'strict', 'guard', 'task'])
def test_bad_identity_or_substitution_rejected(change):
    a, b, pair, c = fixture()
    if change == 'prefix': b['checkpoints'][0]['snapshot'] = 'different'
    if change == 'early': b['own_returned'] = 9
    if change == 'strict': pair['intervention']['final_metrics']['joint_xy_error'] = -1.
    if change == 'guard': pair['safeguard']['selected_index'] = 1
    if change == 'task': b['task_id'] = 'different'
    with pytest.raises(ValueError): m.compare(a, b, pair, c)


def test_model_tie_does_not_imply_native_tie():
    a, b, pair, c = fixture()
    c['evaluations'][2]['cost'] = 2.; pair['safeguard']['selected_Q'] = 2.
    assert m.compare(a, b, pair, c)['new_minus_control']['joint_xy_error'] == -1.
