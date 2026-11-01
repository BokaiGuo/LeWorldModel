import math

from experiments.spectral_reachability.p1_common_h10_analysis_20261101_main import metrics, delta


def test_physical_components_separate_velocity():
    goal = [0, 0, 0, 0, 0, 100, 100]
    result = metrics([3, 4, 0, 2, 2 * math.pi - .1, -100, -100], goal)
    assert result["pusher_xy"] == 5
    assert result["object_xy"] == 2
    assert abs(result["object_angle"] - .1) < 1e-12


def test_difference_direction():
    assert delta({"object_xy": 2}, {"object_xy": 5}) == {"object_xy": -3}
