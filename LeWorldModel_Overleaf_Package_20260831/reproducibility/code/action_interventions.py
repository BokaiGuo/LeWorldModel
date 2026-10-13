"""Deterministic action-sequence interventions shared by replay evaluators."""

from __future__ import annotations

import torch


def derangement_indices(count: int, seed: int) -> list[int]:
    """Return a reproducible permutation with no fixed points."""
    if count < 2:
        raise ValueError("A derangement needs at least two episodes")
    generator = torch.Generator().manual_seed(seed)
    base = torch.arange(count)
    for _ in range(1000):
        permutation = base[torch.randperm(count, generator=generator)]
        if not torch.any(permutation == base):
            return permutation.tolist()
    # A cyclic shift is a deterministic fallback and has no fixed points.
    shift = 1 + seed % (count - 1)
    return torch.roll(base, shifts=shift).tolist()


def per_episode_block_derangement(count: int, seed: int, episode_index: int) -> list[int]:
    """Return a deterministic derangement for one episode's action blocks.

    A distinct derived seed keeps block permutations independent across
    episodes while making the complete intervention exactly reproducible from
    the checkpoint protocol and the two integer seeds.
    """
    derived_seed = int(seed) + 1_000_003 * int(episode_index)
    return derangement_indices(count, derived_seed)


def derange_action_blocks(
    blocks: torch.Tensor,
    seed: int,
    episode_index: int,
    preserve_prefix: int = 0,
) -> tuple[torch.Tensor, list[int]]:
    """Derange future action blocks while preserving an optional history prefix.

    The returned permutation indexes the suffix only.  Thus the intervention
    preserves the exact multiset and marginal values of an episode's future
    action blocks while changing their temporal order.
    """
    if blocks.ndim < 1:
        raise ValueError("blocks must have a block dimension")
    if not 0 <= preserve_prefix < blocks.shape[0]:
        raise ValueError("preserve_prefix must leave at least one block")
    suffix_count = blocks.shape[0] - preserve_prefix
    permutation = per_episode_block_derangement(suffix_count, seed, episode_index)
    suffix = blocks[preserve_prefix:][torch.tensor(permutation, device=blocks.device)]
    if preserve_prefix:
        return torch.cat((blocks[:preserve_prefix], suffix), dim=0), permutation
    return suffix, permutation


def intervene_future_action_blocks(
    blocks: torch.Tensor,
    mode: str,
    preserve_prefix: int,
) -> torch.Tensor:
    """Modify only future action blocks while preserving history exactly."""
    if blocks.ndim < 1:
        raise ValueError("blocks must have a block dimension")
    if not 0 <= preserve_prefix < blocks.shape[0]:
        raise ValueError("preserve_prefix must leave at least one block")
    prefix = blocks[:preserve_prefix]
    future = blocks[preserve_prefix:]
    if mode == "future_zero":
        modified = torch.zeros_like(future)
    elif mode == "future_reverse":
        modified = future.flip(0)
    else:
        raise ValueError(f"unsupported future intervention mode: {mode}")
    return torch.cat((prefix, modified), dim=0)
