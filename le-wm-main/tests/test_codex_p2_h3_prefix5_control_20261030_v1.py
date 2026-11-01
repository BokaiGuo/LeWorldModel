import numpy as np
import pytest

from experiments.spectral_reachability import codex_p2_h3_prefix5_control_20261030_v1 as m


def test_bound_prefix_and_mask():
    binding = m.build()
    original = m.original.build()
    joint = {p["root_index"]: p for p in original["plans"] if p["method"] == "joint"}
    assert [p["root_index"] for p in binding["plans"]] == [0, 1, 2, 3]
    for p in binding["plans"]:
        assert p["commands"][:5] == joint[p["root_index"]]["commands"][:5]
        assert p["commands"][5:] == [[0., 0.]] * 10
        assert np.array_equal(np.frombuffer(bytes.fromhex(p["raw_float32_hex"]), dtype="<f4").reshape(15, 2), p["commands"])
        assert p["original_raw_bits_proved"] is False
    assert binding["preregistered"] is False
    assert binding["budgets"]["selected"] == 60


def test_meter_refuses_step_sixteen():
    class Env:
        def step(self, action):
            return None
    meter = m.Meter(Env(), lambda event: None)
    meter.bind(dict(root_sha256="r", method="joint_prefix5_zero10", selection_sha256="s", planned_commands=15))
    meter.phase = "selected"
    for _ in range(15):
        meter.step([0., 0.])
    with pytest.raises(ValueError):
        meter.step([0., 0.])
