import numpy as np
import pytest

from experiments.spectral_reachability import codex_p1_cp5_tail_rgb_diagnosis_20261101_exclusive as m


def test_physical_metrics_distinguish_agent_and_object():
    value = m.physical([3, 4, 6, 8, 0.2, 0, 0], [0, 0, 0, 0, 0.1, 0, 0])
    assert value["pusher_xy"] == pytest.approx(5)
    assert value["object_xy"] == pytest.approx(10)
    assert value["joint_xy"] == pytest.approx(np.sqrt(125))


def test_only_model_packet_boundaries_are_predictions():
    assert m.prediction_label(5) == "observed"
    assert m.prediction_label(10) == "model_one_packet"
    assert m.prediction_label(15) == "model_two_packets"
    assert m.prediction_label(9) is None
    assert m.prediction_label(11) is None


def test_equal_cp5_requires_full_identity():
    point = {k: "same" for k in m.IDENTITY}
    m.check_cp5(point, point)
    different = dict(point, pixel_sha256="other")
    with pytest.raises(Exception):
        m.check_cp5(point, different)
