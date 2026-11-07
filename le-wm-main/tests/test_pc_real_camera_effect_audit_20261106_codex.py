"""Invariant checks against corrupt physical logs; fixtures are not evidence."""
import numpy as np
import pytest

from experiments.pc_real_camera_effect_audit_20261106_codex import validate


def fixture():
    return {"traces": np.zeros((8, 3, 2, 6, 4)),
            "images": np.broadcast_to(np.uint8(0), (8, 3, 2, 6, 224, 224, 3)),
            "times": np.broadcast_to(np.arange(6) * .05, (8, 3, 2, 6)).copy(),
            "contacts": np.zeros((8, 3, 2, 6))}


def test_valid_clock():
    assert validate(fixture()) == pytest.approx(.05)


def test_view_physics_mismatch_rejected():
    data = fixture()
    data["traces"][0, 0, 1, 3, 0] = .01
    with pytest.raises(AssertionError):
        validate(data)


def test_nonuniform_clock_rejected():
    data = fixture()
    data["times"][0, 0, :, 3] += .01
    with pytest.raises(AssertionError):
        validate(data)


def test_action_initial_state_mismatch_rejected():
    data = fixture()
    data["traces"][0, 1, :, 0, 0] = 1
    with pytest.raises(AssertionError):
        validate(data)
