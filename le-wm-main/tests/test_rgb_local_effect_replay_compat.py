from __future__ import annotations

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from experiments.replay_rgb_local_effect_planning import _rgb_calibrator


def test_replay_ignores_serialized_knn_in_affine_optional_slot() -> None:
    knn_state = {
        "support": torch.zeros((4, 12)),
        "values": torch.zeros((4, 2)),
    }
    assert _rgb_calibrator(knn_state) is None


def test_replay_restores_affine_optional_calibrator() -> None:
    state = {
        "coefficients": np.zeros((3, 2), dtype=np.float32),
        "feature_mean": np.zeros(3, dtype=np.float32),
        "feature_std": np.ones(3, dtype=np.float32),
    }
    assert _rgb_calibrator(state) is not None
