import numpy as np

from experiments.pc_tworoom_dual_rgb_goal_reconstruction_20261107_codex import make_probes


def scene(agent):
    image = np.full((224, 224, 3), 255, dtype=np.uint8)
    image[:, 107:118] = 0
    y, x = np.indices((224, 224))
    image[(x - agent[0]) ** 2 + (y - agent[1]) ** 2 <= 10 ** 2] = [255, 0, 0]
    return image


def test_dual_rgb_recovers_wall_behind_goal_marker():
    current, goal = scene((40, 112)), scene((112, 112))
    probes, info = make_probes(current, goal)
    assert info["valid"] and info["reconstruction_mae"] == 0
    for actual, position in zip(probes, [(116, 112), (108, 112), (112, 116), (112, 108)], strict=True):
        np.testing.assert_array_equal(actual, scene(position))


def test_dual_rgb_abstains_if_marker_hides_same_background():
    image = scene((112, 112))
    probes, info = make_probes(image, image)
    assert probes is None and info["reason"] == "opaque_overlap"


def test_dual_rgb_rejects_changed_background():
    current, goal = scene((40, 112)), scene((112, 112))
    current[10, 10] = [0, 0, 0]
    probes, info = make_probes(current, goal)
    assert probes is None and info["reason"] == "background_mismatch"
