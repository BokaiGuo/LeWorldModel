import hashlib
from io import BytesIO
import json

import numpy as np
import pytest

from experiments.spectral_reachability import (
    p1_dev303_rgb_boundary_static_20261102_exclusive as m,
)


def test_static_diagnostic_verifies_blind_sha_and_zero_actions(monkeypatch):
    history = np.zeros((3, 224, 224, 3), np.uint8)
    goal = np.zeros((224, 224, 3), np.uint8)
    actions = np.zeros((10, 2), np.float32)
    buf = BytesIO()
    np.savez_compressed(buf, rgb_history=history, goal_rgb=goal, raw_action_history=actions)
    raw = buf.getvalue()
    digest = lambda x: hashlib.sha256(x).hexdigest()
    monkeypatch.setattr(m, "NPZ_SHA", digest(raw))
    manifest = {"blind_sha256": digest(raw), "seed": 301000003, "P1_admission": False,
                "independent_confirmation": False, "source_receipt_sha256": "fixture",
                "pixel_sha256": [digest(x.tobytes()) for x in (*history, goal)],
                "action_sha256": digest(actions.tobytes())}
    manifest_raw = json.dumps(manifest).encode()
    monkeypatch.setattr(m, "MANIFEST_SHA", digest(manifest_raw))
    result = m.static_check(manifest_raw, raw, "fixture")
    assert result["result"]["status"] == "abstain"
    assert result["action_history"]["all_zero"] is True
    assert result["source_chain_reverified"] is False
    assert result["P1_admission"] is False
    with pytest.raises(ValueError, match="SHA"):
        m.static_check(manifest_raw, raw + b"x", "fixture")


def test_motion_coefficients_match_frozen_formula():
    assert m.CONFIG["current_action_pixels"] == 29.567 * 224 / 512
    assert m.CONFIG["previous_action_pixels"] == 0.02993 * 374.1 * 224 / 512
