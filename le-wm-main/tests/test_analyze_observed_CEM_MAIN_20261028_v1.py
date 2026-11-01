from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import analyze_observed_CEM_MAIN_20261028_v1 as m


def fixture():
    goal = [0.]*7
    old, refs, rows = [], [], []
    for slot, arm in m.source.ORDER:
        old.append(dict(root_slot=slot, arm=arm, selection_digest='a', goal=goal,
                        final_state=[30.,0.,0.,0.,0.,0.,0.], final_metrics=dict(native_success=False)))
        refs.append(dict(root_slot=slot, arm=arm, original_selection_digest='a', goal=goal,
                         prefix=[dict(state=[50.,0.,0.,0.,0.,0.,0.])]*6))
        rows.append(dict(root_slot=slot, prefix_arm=arm, prefix_exact=True, prefix_goal_exact=True, goal=goal,
                         final_state=[10.,0.,0.,0.,0.,1000.,0.], native_success=True,
                         final_metrics=dict(native_success=True), status='native_success', own_returned=8,
                         terminated=True, truncated=False))
    return rows, refs, old


def test_native_success_uses_joint_xy_and_angle_not_velocity():
    x = m.metrics([10.,0.,0.,0.,0.,1000.,0.], [0.]*7)
    assert x['native_success'] and x['native_state7_distance'] > 1000
    assert not m.metrics([20.,0.,0.,0.,0.,0.,0.], [0.]*7)['native_success']
    assert not m.metrics([0.,0.,0.,0.,np.pi/9,0.,0.], [0.]*7)['native_success']


def test_original_negative_native_angle_edge_not_repaired():
    x = m.metrics([0.,0.,0.,0.,7.,0.,0.], [0.]*7)
    assert x['native_angle_term'] < 0 and x['native_success']


def test_complete_pair_and_native_stop():
    rows, refs, old = fixture(); result = m.paired(rows, refs, old)
    assert result['denominator'] == result['CEM_native_successes'] == 4
    assert result['old_native_successes'] == 0 and result['mean_delta_joint_xy'] == -20
    assert all(r['returned_suffix_steps'] == 3 for r in result['rows'])


@pytest.mark.parametrize('damage', ['order', 'goal', 'digest', 'prefix', 'success', 'denominator', 'failed'])
def test_pair_mutations_rejected(damage):
    rows, refs, old = deepcopy(fixture())
    if damage == 'order': rows.reverse()
    elif damage == 'goal': rows[0]['goal'] = [1.]*7
    elif damage == 'digest': old[0]['selection_digest'] = 'bad'
    elif damage == 'prefix': rows[0]['prefix_exact'] = False
    elif damage == 'success': rows[0]['native_success'] = False
    elif damage == 'denominator': rows.pop()
    else: rows[0]['status'] = 'failed'
    with pytest.raises(ValueError): m.paired(rows, refs, old)
