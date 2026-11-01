from copy import deepcopy

import pytest

from experiments.spectral_reachability import p2_h15_kinodynamic_pairing_20261030_v1 as m


def fixture(n=2):
    goal = [0.0] * 7
    state = [1.0] * 7
    snapshot = {'state': state, 'bodies': {'block': {'velocity': [0, 0]}}}
    attach = dict(branch=0, kind='goal_attached', goal=goal, goal_rgb_sha256='pixels', snapshot=snapshot)
    warm = [dict(branch=0, kind='returned_transition', phase='warmup', step=i,
                 state=state, terminated=False, truncated=False) for i in range(1, 11)]
    plan = [dict(branch=0, kind='returned_transition', phase='plan', step=i,
                 state=state, terminated=False, truncated=False,
                 returned_goal_state=goal, returned_goal_rgb_sha256='pixels') for i in range(1, n+1)]
    observation = dict(native_distance=1.0, object_position_error=2.0, wrapped_object_angle_error=3.0,
                       native_success=False, terminated=False, truncated=False)
    prior = dict(root_slot=0, root_sha256='root', status={x:'horizon_completed' for x in m.four.METHODS},
                 last_step={x:15 for x in m.four.METHODS},
                 timeline=[dict(step=i, **{x:observation for x in m.four.METHODS}) for i in range(1, 16)])
    hrow = dict(root_slot=0, initial_exact=True, root_sha256='root', root_state=state, goal=goal,
                goal_pixel_sha256='pixels')
    krow = dict(root_slot=0, initial_exact=True, arm=m.kino.ARM, root_state=state, goal=goal,
                own_returned=n, status='horizon_completed')
    hp = dict(root_sha256='root', task_id='h', task_sha256='hsha')
    kp = dict(root_slot=0, root_sha256='root', task_id='k', task_sha256='ksha')
    return [prior, hrow, krow, [attach]+warm, deepcopy([attach]+warm+plan), hp, kp]


def test_real_steps_only_and_different_export_ids_allowed():
    result = m.pair(*fixture())
    assert result['common_steps'] == [1, 2]
    assert result['timeline'][2]['kinodynamic'] is None
    assert result['timeline'][2]['kinodynamic_minus_h15'] is None
    assert result['task_exports']['h15_task_id'] != result['task_exports']['beam_task_id']


@pytest.mark.parametrize('damage', ['root', 'goal', 'pixels', 'snapshot', 'warm', 'steps', 'stop', 'drift'])
def test_mismatch_rejected(damage):
    args = fixture()
    if damage == 'root': args[6]['root_sha256'] = 'wrong'
    elif damage == 'goal': args[2]['goal'] = [2.0] * 7
    elif damage == 'pixels': args[4][0]['goal_rgb_sha256'] = 'wrong'
    elif damage == 'snapshot': args[4][0]['snapshot']['bodies']['block']['velocity'] = [1, 0]
    elif damage == 'warm': args[4][1]['state'] = [2.0] * 7
    elif damage == 'steps': args[4][-1]['step'] = 3
    elif damage == 'stop': args[4][-2]['terminated'] = True
    elif damage == 'drift': args[4][-1]['returned_goal_rgb_sha256'] = 'wrong'
    with pytest.raises((ValueError, AssertionError)):
        m.pair(*args)
