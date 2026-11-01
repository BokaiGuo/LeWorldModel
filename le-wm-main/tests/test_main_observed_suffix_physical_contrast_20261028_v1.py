"""Synthetic native7 arithmetic; not physical evidence."""
from copy import deepcopy
import pytest
from experiments.spectral_reachability import main_observed_suffix_physical_contrast_20261028_v1 as m


def fixture():
    a = dict(index=0, root_slot=0, condition='off', task_id='same', root_state=[50., 0, 0, 0, 0, 0, 0],
             goal=[0.]*7, goal_pose=[0.]*3, root_sha256='root', goal_pixel_sha256='goal',
             plan=dict(model_task_binding=dict(task_id='same'), commands=[[0., 0.]]*10),
             checkpoints=[dict(step=5, state=[0.]*7, snapshot={}, file_sha256='file', pixel_sha256='pixels')],
             own_returned=10, native_success=False, final_state=[30., 0, 0, 0, 0, 0, 0],
             final_metrics=dict(native_distance=30.))
    b = deepcopy(a); b.update(prefix_reference_exact=True, native_success=True,
                             final_state=[10., 0, 0, 0, 0, 0, 0], final_metrics=dict(native_distance=10.))
    b['plan'].update(new_suffix_Q=.1, observed_incumbent_Q=.2, commands=[[0., 0.]]*5+[[.1, 0.]]*5)
    return a, b


def test_full_axes_and_model_physical_separate():
    r = m.compare(*fixture())
    assert r['metrics']['delta_new_minus_old']['joint_xy'] == -20
    assert r['metrics']['delta_new_minus_old']['angle_wrapped'] == 0
    assert r['observed_model_Q_delta'] == -.1 and r['new_success']


def test_native_stop_not_padded_or_removed():
    a, b = fixture(); b['own_returned'] = 7
    r = m.compare(a, b)
    assert not r['comparable_step10'] and r['metrics'] is None and r['new_success']


@pytest.mark.parametrize('bad', ['task', 'prefix', 'cpstate', 'png', 'success', 'distance', 'guard'])
def test_physical_identity_or_metric_mutations_rejected(bad):
    a, b = fixture()
    if bad == 'task': b['task_id'] = 'different'
    elif bad == 'prefix': b['plan']['commands'][0] = [.1, 0]
    elif bad == 'cpstate': b['checkpoints'][0]['state'][0] = 1
    elif bad == 'png': b['checkpoints'][0]['pixel_sha256'] = 'different'
    elif bad == 'success': b['native_success'] = False
    elif bad == 'distance': b['final_metrics']['native_distance'] = 9
    else: b['prefix_reference_exact'] = False
    with pytest.raises(ValueError): m.compare(a, b)
