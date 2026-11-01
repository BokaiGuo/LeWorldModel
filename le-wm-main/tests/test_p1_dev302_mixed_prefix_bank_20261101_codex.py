import numpy as np

from experiments.spectral_reachability import p1_dev302_mixed_prefix_bank_20261101_codex as subject


def _image(pusher, obj, goal_obj=None):
    image = np.zeros((224, 224, 3), np.uint8)
    px, py = pusher
    ox, oy = obj
    image[py - 3 : py + 4, px - 3 : px + 4] = subject.rgb_source.rgb.BLUE
    image[oy - 8 : oy + 9, ox - 12 : ox + 13] = subject.rgb_source.rgb.BLOCK
    if goal_obj is not None:
        gx, gy = goal_obj
        image[gy - 8 : gy + 9, gx - 12 : gx + 13] = subject.rgb_source.rgb.BLOCK
    return image


def test_bank_contains_fixed_required_families_without_state():
    current = _image((190, 100), (145, 80))
    goal = _image((120, 70), (170, 180))
    result = subject.build(current, goal, np.zeros((5, 2)), np.ones((5, 2)) * 0.1)
    labels = [row["label"] for row in result["candidates"]]
    assert labels == [
        "zero", "spectral", "gn", "rgb_nearest_contact",
        "rgb_goal_support_direct", "rgb_goal_support_top_route",
        "rgb_goal_support_bottom_route", "random_control",
    ]
    assert result["simulator_state_used"] is False
    assert result["physical_outcome_used"] is False
    assert all(np.asarray(row["actions"]).shape == (5, 2) for row in result["candidates"])
    assert result["candidates"][0]["contact_feasible_prior"] is False
    assert all(result["candidates"][index]["contact_feasible_prior"] for index in (3, 4, 5, 6))


def test_bank_is_deterministic():
    current = _image((190, 100), (145, 80))
    goal = _image((120, 70), (170, 180))
    first = subject.build(current, goal, np.zeros((5, 2)), np.zeros((5, 2)))
    second = subject.build(current, goal, np.zeros((5, 2)), np.zeros((5, 2)))
    assert first == second
