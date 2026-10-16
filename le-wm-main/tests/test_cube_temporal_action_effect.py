import numpy as np
import torch

from experiments.audit_cube_temporal_action_effect import (
    _fit_gated_residual,
    build_feature_families,
)


def test_temporal_feature_family_contains_previous_action_and_invalid_first_frame():
    current = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    previous = current.copy()
    previous[1, 2, 2, 0] = 255
    goals = np.zeros_like(current)
    action = np.zeros((3, 5), dtype=np.float32)
    previous_action = np.zeros_like(action)
    previous_action[2, 0] = 1.0
    valid = np.asarray([0.0, 1.0, 1.0], dtype=np.float32)
    families, event = build_feature_families(
        current, previous, goals, action, previous_action, valid, device="cpu"
    )
    assert families["rgb_temporal"].shape[0] == 3
    assert families["rgb_temporal"].shape[1] > families["rgb_static"].shape[1]
    assert families["rgb_temporal_interaction"].shape[1] > families["rgb_static"].shape[1]
    assert float(event[0]) == 0.0
    assert float(event[1]) > 0.0


def test_event_support_gate_falls_back_when_event_is_absent():
    generator = torch.Generator().manual_seed(4)
    base = torch.randn((20, 4), generator=generator)
    temporal = torch.randn((20, 6), generator=generator)
    target = torch.randn((20, 2), generator=generator)
    event = torch.zeros(20)
    calibration = np.zeros(20, dtype=bool)
    calibration[:10] = True
    test = ~calibration
    gated, details = _fit_gated_residual(
        base, temporal, event, target, calibration, test, ridge=1e-2,
        event_quantile=0.75, support_quantile=0.95,
    )
    assert details["gate_test_activation_rate"] == 0.0
    assert details["metrics"]["test_coverage"] == 0.0
    assert torch.isfinite(gated).all()
