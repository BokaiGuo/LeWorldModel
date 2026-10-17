from __future__ import annotations

import pytest
import torch

from experiments.theory_spectral_transfer.run_jepa_wm_pointmaze_temporal_eval import (
    build_pointmaze_datasets,
    canonical_pooled_latent,
    parse_args,
    rendered_observation,
)


def test_canonical_pooled_latent_preserves_feature_axis_for_token_layouts() -> None:
    token_first = torch.arange(16 * 384, dtype=torch.float32).reshape(16, 384)
    channel_first = token_first.transpose(0, 1)
    assert canonical_pooled_latent(token_first).shape == (384,)
    assert torch.allclose(canonical_pooled_latent(token_first), token_first.mean(dim=0))
    assert torch.allclose(canonical_pooled_latent(channel_first), channel_first.mean(dim=1))


def test_canonical_pooled_latent_fails_closed_on_ambiguous_or_wrong_shape() -> None:
    with pytest.raises(ValueError, match="unique feature axis"):
        canonical_pooled_latent(torch.zeros(16, 32))
    with pytest.raises(ValueError, match="unique feature axis"):
        canonical_pooled_latent(torch.zeros(384, 384))


def test_pointmaze_export_leaves_visual_transform_to_model_encode(tmp_path) -> None:
    calls = []

    class FakeDataset:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.transform = kwargs.get("transform")

    stats, observations = build_pointmaze_datasets(FakeDataset, tmp_path)

    assert stats is not observations
    assert calls == [
        {"data_path": str(tmp_path), "normalize_action": True},
        {
            "data_path": str(tmp_path),
            "transform": None,
            "normalize_action": False,
        },
    ]
    assert observations.transform is None


def test_rendered_observation_stays_raw_and_uses_selected_states() -> None:
    class FakeRenderer:
        def render(self, state):
            value = int(state[0])
            return torch.full((4, 5, 3), value, dtype=torch.uint8).numpy()

    states = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]]
    )
    observation = rendered_observation(FakeRenderer(), states, [2, 1])

    assert observation["visual"].shape == (2, 3, 4, 5)
    assert observation["visual"].dtype == torch.float32
    assert observation["visual"][0].unique().tolist() == [20.0]
    assert torch.equal(observation["proprio"], states[[2, 1]])


def test_renderer_calibration_frame_budget_matches_online_default(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "pointmaze-export",
            "--dataset-root", "dataset",
            "--checkpoint", "checkpoint.pt",
            "--output", "output.json",
            "--visual-source", "dataset_matched_renderer",
        ],
    )

    args = parse_args()

    assert args.renderer_calibration_frames == 768
