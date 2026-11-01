"""Pure numerical fixtures, not model/physics evidence."""
from copy import deepcopy
import pytest
from test_main_observed_suffix_physical_contrast_20261028_v1 import fixture
from experiments.spectral_reachability import main_observed_suffix_rgb_contrast_20261028_v1 as m


def test_reversal_and_signed_cancellation_preserved():
    old, new = fixture()
    r = m.contrast([2.]*192, [1.]*192, [.5]*192, [.8]*192, [0.]*192, old, new)
    assert r['predicted_delta'] == -3 and r['actual_delta'] > 0 and r['ranking_reversed']
    assert abs(sum(r['signed_error_terms'].values()) - r['gap']) < 1e-12
    assert r['native']['metrics']['delta_new_minus_old']['joint_xy'] == -20
    assert not r['exact_action_independent_offset']


def test_exact_common_offset_can_hold_without_assuming_it():
    r = m.contrast([2.]*192, [1.]*192, [1.5]*192, [.5]*192, [0.]*192, *fixture())
    assert r['exact_action_independent_offset'] and r['response_error_MSE'] == 0


@pytest.mark.parametrize('bad', ['shape', 'nan', 'task', 'cp5'])
def test_bad_goal_or_native_binding_rejected(bad):
    old, new = deepcopy(fixture()); goal = [0.]*192
    if bad == 'shape': goal.pop()
    elif bad == 'nan': goal[0] = float('nan')
    elif bad == 'task': new['task_id'] = 'other'
    else: new['checkpoints'][0]['pixel_sha256'] = 'other'
    with pytest.raises(ValueError): m.contrast([2.]*192, [1.]*192, [.5]*192, [.8]*192, goal, old, new)
