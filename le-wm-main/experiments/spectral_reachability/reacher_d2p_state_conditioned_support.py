"""Frozen-latent, state-conditioned official-source proposal support."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability.reacher_d2p_packet_support import (
    SOURCE_SUPPORTED_PROPOSAL_DISTRIBUTION, load_history_packet_pool, receipt_source,
)

STATE_CONDITIONED_SOURCE_PROPOSAL_DISTRIBUTION = "source_locked_frozen_latent_state_conditioned_packet_support"


def is_state_conditioned(distribution: str) -> bool:
    return distribution == STATE_CONDITIONED_SOURCE_PROPOSAL_DISTRIBUTION


def is_source_supported(distribution: str) -> bool:
    return distribution in {SOURCE_SUPPORTED_PROPOSAL_DISTRIBUTION, STATE_CONDITIONED_SOURCE_PROPOSAL_DISTRIBUTION}


def _anchor_rows(lengths: np.ndarray, offsets: np.ndarray, *, count: int, seed: int) -> np.ndarray:
    candidates = np.flatnonzero(np.asarray(lengths, dtype=np.int64) >= 21)
    if count < 1 or not len(candidates):
        raise ValueError("state-conditioned source index has no valid causal anchors")
    rng = np.random.default_rng(seed)
    episodes = rng.choice(candidates, size=count, replace=len(candidates) < count)
    return np.asarray([
        int(rng.integers(int(offsets[e]) + 15, int(offsets[e]) + int(lengths[e]) - 5))
        for e in episodes.tolist()
    ], dtype=np.int64)


def _nearest(query: np.ndarray, anchors: np.ndarray, *, count: int, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    query, anchors = np.asarray(query, dtype=np.float64), np.asarray(anchors, dtype=np.float64)
    mean, scale = np.asarray(mean, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    if query.ndim != 1 or anchors.ndim != 2 or anchors.shape[1] != query.size or mean.shape != query.shape or scale.shape != query.shape or np.any(scale <= 0.0) or count < 1 or count > len(anchors):
        raise ValueError("state-conditioned source nearest-neighbor request is incompatible")
    distance = np.square((anchors - query[None]) / scale[None]).mean(axis=1)
    return np.lexsort((np.arange(len(distance)), distance))[:count].astype(np.int64, copy=False)


def build_support(*, adapter: Path, model: torch.nn.Module, preprocessor: Any, device: torch.device,
                  history_len: int, action_block: int, index_size: int, index_seed: int, batch_size: int) -> dict[str, Any]:
    """Build deterministic source anchors before any evaluator root is executed."""
    dataset, digest = receipt_source(adapter)
    with h5py.File(dataset, "r", swmr=True) as handle:
        rows = _anchor_rows(np.asarray(handle["ep_len"]), np.asarray(handle["ep_offset"]), count=index_size, seed=index_seed)
    embeddings, packets = [], []
    with h5py.File(dataset, "r", swmr=True) as handle:
        for start in range(0, len(rows), batch_size):
            selected = rows[start:start + batch_size]
            images = np.stack([np.asarray(handle["pixels"][[r - 10, r - 5, r]], dtype=np.uint8) for r in selected])
            history = np.stack([np.asarray(handle["action"][r - 15:r], dtype=np.float32).reshape(history_len, -1) for r in selected])
            packed = preprocessor._prepare_info({"pixels": images, "action": history})
            with torch.no_grad():
                encoded = p0._latent_context(model, packed["pixels"].to(device), packed["action"].to(device))[:, -1]
            embeddings.append(encoded.detach().cpu().numpy().astype(np.float32, copy=False))
            packets.append(np.stack([np.asarray(handle["action"][r:r + action_block], dtype=np.float32) for r in selected]))
    histories, history_support = load_history_packet_pool(dataset, expected_sha256=digest, history_len=history_len, action_block=action_block, verify_source=False)
    anchor_embeddings = np.concatenate(embeddings)
    feature_mean = anchor_embeddings.mean(axis=0)
    feature_scale = anchor_embeddings.std(axis=0).clip(1.0e-6)
    return {"embeddings": anchor_embeddings, "feature_mean": feature_mean, "feature_scale": feature_scale, "packets": np.concatenate(packets), "histories": histories,
            "receipt": {"dataset": str(dataset), "dataset_sha256": digest, "index_size": int(index_size), "index_seed": int(index_seed), "index_rows_sha256": hashlib.sha256(rows.tobytes()).hexdigest(), "batch_size": int(batch_size), "distribution": STATE_CONDITIONED_SOURCE_PROPOSAL_DISTRIBUTION, "history_support": history_support}}


def proposal_bank(*, support: dict[str, Any], pixels: torch.Tensor, history: torch.Tensor, model: torch.nn.Module,
                  proposals: int, mean: np.ndarray, std: np.ndarray, lower: torch.Tensor, upper: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    if proposals < 2:
        raise ValueError("state-conditioned support needs raw-zero plus at least one source proposal")
    query = p0._latent_context(model, pixels, history)[0, -1].detach().cpu().numpy()
    indices = _nearest(query, support["embeddings"], count=proposals - 1, mean=support["feature_mean"], scale=support["feature_scale"])
    raw = np.asarray(support["packets"][indices], dtype=np.float32).reshape(proposals - 1, 1, -1)
    values = torch.as_tensor(raw, device=lower.device)
    normalized = (values - torch.as_tensor(mean, device=lower.device)) / torch.as_tensor(std, device=lower.device)
    coordinate = (2.0 * normalized - (upper + lower)) / (upper - lower)
    eps = torch.finfo(values.dtype).eps
    if not torch.isfinite(coordinate).all() or torch.any(coordinate <= -1.0) or torch.any(coordinate >= 1.0):
        raise ValueError("state-conditioned source packet leaves deployed action chart")
    bank = torch.cat((torch.zeros((1, 1, history.shape[-1]), device=lower.device), torch.atanh(coordinate.clamp(-1.0 + eps, 1.0 - eps))), dim=0)
    return bank, {"proposal_distribution": STATE_CONDITIONED_SOURCE_PROPOSAL_DISTRIBUTION + "_with_raw_zero_at_index_0", "retrieval_metric": "source_anchor_per_dimension_zscore_l2", "retrieved_source_index_rows": indices.tolist(), "index_rows_sha256": support["receipt"]["index_rows_sha256"]}
