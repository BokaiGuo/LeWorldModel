import numpy as np

from experiments.run_pusht_receding_block_pose_planner import make_state_only_env
from experiments.run_pusht_signed_torque_audit import (
    boundary_inward_normal,
    make_terminal_roots,
    physical_contact_geometry,
)


def test_signed_torque_radial_roots_use_polygon_boundary_contact():
    env = make_state_only_env()
    try:
        roots = make_terminal_roots(
            env,
            8,
            20261014,
            root_protocol="radial_contact",
            support_gap_max=0.35,
        )
        distances = [physical_contact_geometry(env, root["state"])["distance"] for root in roots]
        assert len(roots) == 8
        assert np.all(np.isfinite(distances))
        assert max(distances) <= 24.0
        assert min(distances) >= 10.0
        for root in roots:
            direction = boundary_inward_normal(env, root["state"])
            assert np.isclose(np.linalg.norm(direction), 1.0, atol=1e-6)
            assert np.isfinite(direction).all()
    finally:
        env.close()
