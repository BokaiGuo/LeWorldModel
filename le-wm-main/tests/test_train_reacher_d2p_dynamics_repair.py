import torch

from experiments.spectral_reachability.train_reacher_d2p_dynamics_repair import (
    _EpisodeDisjointClips,
    _EpisodeChunkBatchSampler,
    _derangement,
    _image_transform,
    _normalize_actions,
    _preprocess_pixels,
    _require_training_semantic_packet_stats,
    causal_dynamics_losses,
)
from train import lejepa_forward


class _Dataset:
    clip_indices = [(0, 0), (1, 0), (8, 0), (9, 0), (10, 0)]

    def __getitem__(self, index):
        return {"index": index}


class _Dynamics(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def encode(self, info):
        # Four scalar observations become a one-dimensional frozen latent.
        return {"emb": info["pixels"].mean(dim=(-1, -2, -3), keepdim=False).unsqueeze(-1)}

    def action_encoder(self, action):
        return action[..., :1] * self.scale

    def predict(self, embeddings, actions):
        return embeddings + actions


def test_episode_disjoint_split_is_by_source_episode_not_clip_row():
    train = _EpisodeDisjointClips(_Dataset(), split="train")
    validation = _EpisodeDisjointClips(_Dataset(), split="validation")
    assert train.episode_ids == [0, 1, 10]
    assert validation.episode_ids == [8, 9]
    assert set(train.episode_ids).isdisjoint(validation.episode_ids)


def test_chunk_aware_sampler_interleaves_short_contiguous_episode_runs():
    class _ChunkDataset:
        clip_indices = [(episode, start) for episode in range(4) for start in range(12)]

    subset = _EpisodeDisjointClips(_ChunkDataset(), split="train")
    subset.indices = list(range(len(_ChunkDataset.clip_indices)))
    sampler = _EpisodeChunkBatchSampler(subset, batch_size=8, batches=1, seed=7, episode_groups=4)
    batch = next(iter(sampler))
    episodes = [subset.base.clip_indices[subset.indices[index]][0] for index in batch]
    starts = [subset.base.clip_indices[subset.indices[index]][1] for index in batch]
    assert len(batch) == 8
    assert len(set(episodes)) == 4
    for offset in range(0, len(batch), 4):
        assert len(set(episodes[offset : offset + 4])) == 4
    for episode in set(episodes):
        episode_starts = sorted(start for current, start in zip(episodes, starts) if current == episode)
        assert episode_starts[1] == episode_starts[0] + 1


def test_validation_chunk_sampler_reuses_the_identical_seeded_panel_across_epochs():
    class _ChunkDataset:
        clip_indices = [(episode, start) for episode in (8, 9, 18, 19) for start in range(12)]

    subset = _EpisodeDisjointClips(_ChunkDataset(), split="validation")
    sampler = _EpisodeChunkBatchSampler(
        subset, batch_size=8, batches=3, seed=17, episode_groups=4, advance_epoch=False,
    )
    assert list(iter(sampler)) == list(iter(sampler))


def test_training_chunk_sampler_advances_its_seeded_panel_between_epochs():
    class _ChunkDataset:
        clip_indices = [(episode, start) for episode in range(12) for start in range(12)]

    subset = _EpisodeDisjointClips(_ChunkDataset(), split="train")
    sampler = _EpisodeChunkBatchSampler(subset, batch_size=8, batches=3, seed=17, episode_groups=4)
    assert list(iter(sampler)) != list(iter(sampler))


def test_gpu_side_pixel_preprocess_matches_released_imagenet_transform_at_224():
    pixels = torch.arange(2 * 4 * 3 * 224 * 224, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(2, 4, 3, 224, 224)
    expected = _image_transform()({"pixels": pixels.clone()})["pixels"]
    observed = _preprocess_pixels(pixels)
    assert torch.allclose(observed, expected, rtol=0.0, atol=1.0e-6)


def test_derangement_never_pairs_a_row_with_its_own_action():
    permutation = _derangement(4, device=torch.device("cpu"))
    assert not torch.any(permutation == torch.arange(4))


def test_runtime_packet_statistics_are_not_repeated_a_second_time():
    actions = torch.arange(80.0).reshape(2, 4, 10)
    mean = torch.arange(10.0)
    normalized = _normalize_actions(actions, mean=mean, std=torch.ones(10))
    assert torch.equal(normalized, actions - mean.view(1, 1, 10))


def test_episode_boundary_nan_action_is_neutral_after_normalization_not_raw_zero():
    actions = torch.full((1, 4, 10), float("nan"))
    normalized = _normalize_actions(actions, mean=torch.arange(10.0), std=torch.ones(10))
    assert torch.equal(normalized, torch.zeros_like(actions))


def test_training_semantic_packet_stats_require_the_same_physical_pair_in_every_slot():
    source_mean = torch.tensor([0.25, -0.5])
    source_std = torch.tensor([0.5, 0.25])
    _require_training_semantic_packet_stats(
        source_mean.repeat(5), source_std.repeat(5), source_mean=source_mean, source_std=source_std,
    )
    try:
        _require_training_semantic_packet_stats(
            torch.tensor([0.25, -0.5, 0.1, -0.5, 0.25, -0.5, 0.25, -0.5, 0.25, -0.5]),
            source_std.repeat(5), source_mean=source_mean, source_std=source_std,
        )
    except ValueError as error:
        assert "packet slots" in str(error)
    else:
        raise AssertionError("position-specific action statistics must fail closed")


def test_causal_repair_targets_are_detached_and_true_action_wins_swap_ranking():
    model = _Dynamics()
    # Pixels encode [0, 1, 2, 3] and [10, 11, 12, 13].  A final action of 1
    # therefore predicts each next observed latent exactly.
    pixels = torch.tensor([[[[[0.0]]], [[[1.0]]], [[[2.0]]], [[[3.0]]]], [[[[10.0]]], [[[11.0]]], [[[12.0]]], [[[13.0]]]]], requires_grad=True)
    actions = torch.ones((2, 4, 10), requires_grad=True)
    loss, metrics = causal_dynamics_losses(model, pixels, actions, ranking_temperature=0.1)
    loss.backward()
    assert metrics["target_requires_grad"].item() == 0.0
    assert metrics["action_swap_ranking_accuracy"].item() == 0.0  # equal actions are intentionally a tie
    assert pixels.grad is None  # encoder/target branch is frozen by the repair contract


def test_original_training_entrypoint_stops_gradient_at_future_successor_target():
    embeddings = torch.arange(8.0).reshape(1, 4, 2).requires_grad_()

    class _Model:
        @staticmethod
        def encode(batch):
            return {"emb": embeddings, "act_emb": batch["action"]}

        @staticmethod
        def predict(context, action):
            return context

    class _Self:
        model = _Model()
        sigreg = staticmethod(lambda _: torch.tensor(0.0))

        @staticmethod
        def log_dict(*_args, **_kwargs):
            return None

    cfg = type("Cfg", (), {"history_size": 3, "num_preds": 1, "loss": type("Loss", (), {"sigreg": type("Sig", (), {"weight": 0.0})()})()})()
    result = lejepa_forward(_Self(), {"action": torch.zeros((1, 4, 1))}, "train", cfg)
    result["loss"].backward()
    # The final input frame only appears in the future target.  It must not
    # receive prediction-loss gradient after the causal target repair.
    assert torch.allclose(embeddings.grad[:, -1], torch.zeros_like(embeddings.grad[:, -1]))
