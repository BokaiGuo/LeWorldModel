import numpy as np
import pytest
from experiments.spectral_reachability import p5_native_contact_geometry_main_20261027_v1 as m


@pytest.mark.parametrize('reverse', [False, True])
def test_outside_polygon_gap(reverse):
    vertices = np.array([[0,0], [2,0], [2,2], [0,2]], float)
    if reverse:
        vertices = vertices[::-1]
    assert m.polygon_gap([4,1], vertices, 1) == 1
    assert np.isclose(m.polygon_gap([3,3], vertices, 1), np.sqrt(2)-1)


def test_reject_inside_union_shortcut():
    with pytest.raises(ValueError):
        m.polygon_gap([1,1], [[0,0],[2,0],[2,2],[0,2]], 1)


def test_world_position_not_cog():
    body = dict(angle=np.pi/2, position=[10,20], center_of_gravity=[0,45])
    assert np.allclose(m.world_vertices([[1,0]], body), [[10,21]])


def test_zero_command_still_moves_with_velocity():
    points, velocity = m.pd_substeps([0,0], [1,0], [0,0])
    assert points.shape == (10,2) and points[0,0] == .008
    assert np.any(points != 0) and np.isfinite(velocity).all()


def test_zero_state_zero_command_stationary():
    points, velocity = m.pd_substeps([1,2], [0,0], [0,0])
    assert np.array_equal(points, np.tile([1.,2.], (10,1)))
    assert np.array_equal(velocity, [0,0])
