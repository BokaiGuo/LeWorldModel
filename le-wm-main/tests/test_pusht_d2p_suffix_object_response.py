import numpy as np
import pytest

from experiments.spectral_reachability.audit_pusht_d2p_suffix_object_response import response


def branch(after):
    return dict(successor_state=[0., 0., 20., 30., .5, 0., 0.], final_state=after,
                prefix_pose={"pose_cost": 10.}, final_pose={"pose_cost": 10.})


def test_agent_motion_is_not_block_motion():
    result = response(branch([5., 2., 20., 30., .5, 1., 1.]))
    assert result["block_pose_exactly_unchanged"]
    assert result["agent_xy_displacement"] == pytest.approx(np.sqrt(29))
    assert result["block_xy_displacement"] == 0


def test_angle_wrap_is_reported_without_exact_identity_claim():
    result = response(branch([0., 0., 20., 30., .5 + 2*np.pi, 0., 0.]))
    assert result["block_angle_displacement_abs"] < 1e-14
    assert not result["block_pose_exactly_unchanged"]


@pytest.mark.parametrize("after", [[0.] * 6, [np.nan] * 7, [np.inf] * 7])
def test_invalid_states_rejected(after):
    with pytest.raises(ValueError):
        response(branch(after))
