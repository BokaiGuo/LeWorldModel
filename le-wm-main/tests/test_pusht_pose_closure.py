import numpy as np

from experiments.run_pusht_pose_closure import pose


def test_pose_excludes_velocity_coordinates():
    np.testing.assert_array_equal(pose(np.arange(8.0)), np.arange(6.0))
