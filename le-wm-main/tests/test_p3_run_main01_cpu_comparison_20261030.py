import math

import pytest

from experiments.spectral_reachability.p3_run_main01_cpu_comparison_20261030 import digest, metrics


def test_native_metric_keeps_velocity_and_native_angle():
    m = metrics([3, 4, 0, 0, 0, 6, 8], [0, 0, 0, 0, 0, 0, 0])
    assert m["joint_xy"] == 5
    assert m["velocity_l2"] == 10
    assert m["native7_distance"] == pytest.approx(math.sqrt(125))
    assert m["native_success"]
    assert metrics([0, 0, 0, 0, 7, 0, 0], [0] * 7)["native_angle_term"] == pytest.approx(2 * math.pi - 7)


def test_invalid_state_rejected():
    with pytest.raises(ValueError):
        metrics([float("nan")] * 7, [0] * 7)
    with pytest.raises(ValueError):
        metrics([0], [0] * 7)


def test_sha_binding(tmp_path):
    path = tmp_path / "input"
    path.write_bytes(b"abc")
    assert digest(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
