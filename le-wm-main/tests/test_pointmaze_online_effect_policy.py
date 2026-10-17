from __future__ import annotations

import numpy as np
import torch

from experiments.decision_benchmark.fit_pointmaze_online_effect_policy import (
    OnlineEffectPolicy,
    fit_nearest_support,
    support_distance,
)
from experiments.decision_benchmark.train_candidate_effect_adapter import EffectMLP


def _zero_member(input_dim: int) -> tuple[EffectMLP, dict[str, object]]:
    model = EffectMLP(input_dim)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    metadata = {
        "input_dim": input_dim,
        "x_mean": [0.0] * input_dim,
        "x_scale": [1.0] * input_dim,
        "y_mean": [0.0, 0.0],
        "y_scale": [1.0, 1.0],
    }
    return model, metadata


def test_nearest_support_accepts_training_point_and_rejects_far_query() -> None:
    values = np.asarray([[0.0, 0.0], [1.0, 0.5], [-0.5, 1.0], [0.75, -1.0]])
    model = fit_nearest_support(values)

    assert support_distance(model, values[0]) == 0.0
    assert support_distance(model, np.asarray([100.0, -100.0])) > model["threshold"]


def test_online_effect_policy_scores_repeated_pulse_candidates() -> None:
    input_dim = 2 + 16
    model, metadata = _zero_member(input_dim)
    latent_support = fit_nearest_support(
        np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    )
    action_support = fit_nearest_support(np.asarray([
        np.linspace(-1.0, 1.0, 16),
        np.linspace(1.0, -1.0, 16),
        np.zeros(16),
        np.ones(16),
    ]))
    policy = OnlineEffectPolicy(
        models=[model],
        normalizations=[metadata],
        pca_mean=np.zeros(2),
        pca_basis=np.eye(2),
        latent_support=latent_support,
        action_support=action_support,
        regret_calibration={"status": "ok", "bound": 0.1},
        action_steps=20,
        feature_mode="frozen_latent_action_basis",
        history_windows=1,
        device=torch.device("cpu"),
    )

    report = policy.score(
        np.asarray([0.0, 0.0]),
        np.asarray([0.25, 0.5]),
        np.asarray([1.0, 1.0]),
        np.asarray([[0.5, 0.0], [0.0, 0.5]]),
    )

    assert np.asarray(report["costs"]).shape == (2,)
    assert np.asarray(report["effects"]).shape == (2, 2)
    assert np.allclose(report["effects"], 0.0)
    assert np.allclose(report["terminals"], [[0.25, 0.5], [0.25, 0.5]])
    assert np.asarray(report["action_support_distances"]).shape == (2,)
    assert report["latent_support_distance"] == 0.0


def test_online_effect_policy_loads_legacy_single_frame_v1(tmp_path) -> None:
    input_dim = 2 + 16
    model, metadata = _zero_member(input_dim)
    support = fit_nearest_support(
        np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    )
    action_support = fit_nearest_support(np.asarray([
        np.linspace(-1.0, 1.0, 16), np.linspace(1.0, -1.0, 16),
        np.zeros(16), np.ones(16),
    ]))
    path = tmp_path / "legacy.pt"
    torch.save({
        "schema": "pointmaze_online_effect_policy.v1",
        "ensemble": [{"state_dict": model.state_dict(), "normalization": metadata}],
        "pca_mean": [0.0, 0.0],
        "pca_basis": np.eye(2).tolist(),
        "latent_support": support,
        "action_support": action_support,
        "regret_calibration": {"status": "ok", "bound": 0.1},
        "action_steps": 20,
    }, path)

    loaded = OnlineEffectPolicy.load(path, torch.device("cpu"))

    assert loaded.feature_mode == "frozen_latent_action_basis"
    assert loaded.history_windows == 1
