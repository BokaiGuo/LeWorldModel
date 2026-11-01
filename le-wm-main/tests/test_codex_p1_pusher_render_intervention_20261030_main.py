import numpy as np

from experiments.spectral_reachability.codex_p1_pusher_render_intervention_20261030_main import OFFSETS, images


def test_fixed_offsets_and_reconstructed_goal_frames():
    assert len(OFFSETS) == len(set(OFFSETS)) == 13
    _, rows = images()
    assert len(rows) == 52
    for root in (0, 1):
        for condition in ("off", "on"):
            group = [r for r in rows if r["root_slot"] == root and r["condition"] == condition]
            assert [tuple(r["offset"]) for r in group] == list(OFFSETS)
            assert np.array_equal(group[0]["image"], group[0]["goal"])
            assert all(r["object_pose"] == group[0]["object_pose"] for r in group)
