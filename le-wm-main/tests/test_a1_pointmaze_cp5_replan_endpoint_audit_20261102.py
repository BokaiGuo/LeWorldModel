import json

import numpy as np
import pytest
import torch

from experiments.a0_pointmaze_native_render_probe_20261102 import DATA
from experiments.a1_pointmaze_cp5_replan_endpoint_audit_20261102 import (
    OUTPUT, SOURCE, SOURCE_SHA, digest, make_sequences,
)


def test_fixed_panel_actions_bind_source():
    report = json.loads(SOURCE.read_text())
    states = torch.load(DATA / "states.pth", map_location="cpu", weights_only=True).float().numpy()
    starts, actions, roster = make_sequences(report, states)
    assert starts.shape == (16, 4)
    assert actions.shape == (16, 10, 2)
    for action, (_, branch) in zip(actions, roster):
        np.testing.assert_array_equal(action[:5], np.repeat([branch["prefix_action"]], 5, axis=0))
        np.testing.assert_array_equal(action[5:], np.repeat([branch["tail_action"]], 5, axis=0))
    report["rows"][0]["branches"][0]["prefix_action"] = [float("nan"), 0]
    with pytest.raises(ValueError):
        make_sequences(report, states)


def test_saved_endpoint_arithmetic():
    if not OUTPUT.exists():
        pytest.skip("audit not yet run")
    result = json.loads(OUTPUT.read_text())
    assert result["source_sha256"] == digest(SOURCE) == SOURCE_SHA
    assert len(result["rows"]) == 16
    for row in result["rows"]:
        a, b = np.asarray(row["legacy_state"]), np.asarray(row["position_backend_state"])
        assert np.linalg.norm(a[:2] - b[:2]) == pytest.approx(row["position_delta_l2"])
        assert np.linalg.norm(a[2:] - b[2:]) == pytest.approx(row["velocity_delta_l2"])
