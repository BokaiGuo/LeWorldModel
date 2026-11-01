"""CUDA diagnostic for state-conditioned, source-supported Reacher P1 actions.

The prior source-supported P1 bank sampled legal action packets marginally
from the training archive.  A packet can therefore be individually in-source
while its pairing with the current live root never occurred in the archive.
This diagnostic forms a fixed retrieval index of causal source contexts using
the frozen encoder, retrieves nearest source contexts for each live root, and
uses only their observed next five-action packets as P1 proposals.  No qpos is
available to the planner; evaluator qpos is recorded only after execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np
import torch

from experiments.audit_live_reacher_shared_candidate_ranking import _Preprocessor
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from experiments.run_live_reacher_planning import ReacherGoalImageWrapper, _load_action_stats, install_missing_field_compat
from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability import run_reacher_d2p_p1_reconfiguration as runner
from experiments.spectral_reachability.preflight_reacher_d2p_p0 import _normalized_action_bounds
from experiments.spectral_reachability.reacher_d2p_context import collect_executed_context
from experiments.spectral_reachability.reacher_d2p_packet_support import (
    WARMUP_POLICY, load_history_packet_pool, receipt_source, sample_history_packets,
)
from experiments.spectral_reachability.reacher_d2p_qpos_readout import qpos_readout_binding, with_qpos_readout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_anchor_rows(lengths: np.ndarray, offsets: np.ndarray, *, count: int, seed: int) -> np.ndarray:
    """Sample source t with [t-10,t-5,t] history and t:t+5 successor intact."""

    if count < 1:
        raise ValueError("state-conditioned source index must have a positive size")
    candidates = np.flatnonzero(np.asarray(lengths, dtype=np.int64) >= 21)
    if not len(candidates):
        raise ValueError("official source has no episode long enough for a causal source anchor")
    rng = np.random.default_rng(seed)
    episodes = rng.choice(candidates, size=count, replace=len(candidates) < count)
    rows = []
    for episode in episodes.tolist():
        start, length = int(offsets[episode]), int(lengths[episode])
        rows.append(int(rng.integers(start + 15, start + length - 5)))
    return np.asarray(rows, dtype=np.int64)


def _nearest_source_indices(query: np.ndarray, anchors: np.ndarray, *, count: int) -> np.ndarray:
    """Return a deterministic frozen-latent nearest-neighbor proposal set."""

    value, bank = np.asarray(query, dtype=np.float64), np.asarray(anchors, dtype=np.float64)
    if value.ndim != 1 or bank.ndim != 2 or bank.shape[1] != value.size or not np.isfinite(value).all() or not np.isfinite(bank).all():
        raise ValueError("state-conditioned support has incompatible non-finite latent features")
    if count < 1 or count > len(bank):
        raise ValueError("state-conditioned support requested an invalid neighbor count")
    distances = np.square(bank - value[None]).mean(axis=1)
    return np.lexsort((np.arange(len(distances)), distances))[:count].astype(np.int64, copy=False)


def _raw_packets_to_params(
    packets: np.ndarray, *, mean: np.ndarray, std: np.ndarray, lower: torch.Tensor, upper: torch.Tensor,
) -> torch.Tensor:
    values = torch.as_tensor(np.asarray(packets, dtype=np.float32).reshape(len(packets), 1, -1), device=lower.device)
    normalized = (values - torch.as_tensor(mean, device=lower.device)) / torch.as_tensor(std, device=lower.device)
    coordinate = (2.0 * normalized - (upper + lower)) / (upper - lower)
    epsilon = torch.finfo(values.dtype).eps
    if not torch.isfinite(coordinate).all() or torch.any(coordinate <= -1.0) or torch.any(coordinate >= 1.0):
        raise ValueError("retrieved source packet leaves the deployed action chart")
    return torch.atanh(coordinate.clamp(-1.0 + epsilon, 1.0 - epsilon))


def _build_source_index(
    *, dataset: Path, rows: np.ndarray, model: torch.nn.Module, preprocessor: _Preprocessor,
    device: torch.device, batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode causal official contexts and retain their observed next packets."""

    embeddings: list[np.ndarray] = []
    packets: list[np.ndarray] = []
    with h5py.File(dataset, "r", swmr=True) as handle:
        for start in range(0, len(rows), batch_size):
            selected = rows[start:start + batch_size]
            images = np.stack([np.asarray(handle["pixels"][[row - 10, row - 5, row]], dtype=np.uint8) for row in selected])
            history = np.stack([np.asarray(handle["action"][row - 15:row], dtype=np.float32).reshape(3, -1) for row in selected])
            packed = preprocessor._prepare_info({"pixels": images, "action": history})
            with torch.no_grad():
                encoded = p0._latent_context(model, packed["pixels"].to(device), packed["action"].to(device))[:, -1]
            embeddings.append(encoded.detach().cpu().numpy().astype(np.float32, copy=False))
            packets.append(np.stack([np.asarray(handle["action"][row:row + 5], dtype=np.float32) for row in selected]))
    return np.concatenate(embeddings), np.concatenate(packets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--qpos-readout", type=Path, required=True)
    parser.add_argument("--predicted-qpos-readout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=8)
    parser.add_argument("--proposals", type=int, default=64)
    parser.add_argument("--source-index-size", type=int, default=4096)
    parser.add_argument("--source-index-seed", type=int, default=2026103700)
    parser.add_argument("--seed", type=int, default=2026103800)
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("state-conditioned support audit requires host CUDA exactly; CPU fallback is forbidden")
    if min(args.roots, args.proposals, args.source_index_size, args.batch_size) < 1 or args.proposals < 2:
        raise ValueError("state-conditioned support requires positive roots/index/batch and at least two proposals")
    output, policy, adapter = args.output.resolve(), args.policy.resolve(), args.adapter_artifact.resolve()
    if output.exists():
        raise FileExistsError("state-conditioned support output must be a new immutable path")
    install_missing_field_compat(); device = torch.device("cuda"); torch.empty(0, device=device)
    mean, std = _load_action_stats(adapter, "swm/ReacherDMControl-v0")
    dataset, dataset_sha256 = receipt_source(adapter)
    binding = qpos_readout_binding(args.qpos_readout, args.predicted_qpos_readout,
        policy_weights_sha256=_sha256(policy / "weights.pt"), adapter_sha256=_sha256(adapter))
    base = load_compatible_pretrained(policy, device="cuda").eval(); base.requires_grad_(False); base.interpolate_pos_encoding = True
    model = with_qpos_readout(base, binding).to(device).eval()
    preprocessor = _Preprocessor(ActionNormalizer(mean, std))
    with h5py.File(dataset, "r", swmr=True) as handle:
        source_rows = _source_anchor_rows(
            np.asarray(handle["ep_len"]), np.asarray(handle["ep_offset"]),
            count=args.source_index_size, seed=args.source_index_seed,
        )
    source_embeddings, source_packets = _build_source_index(dataset=dataset, rows=source_rows, model=model, preprocessor=preprocessor, device=device, batch_size=args.batch_size)
    histories, history_support = load_history_packet_pool(dataset, expected_sha256=dataset_sha256, history_len=3, verify_source=False)
    env = ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65))
    lower_np, upper_np = _normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    lower, upper = torch.as_tensor(lower_np, device=device), torch.as_tensor(upper_np, device=device)
    rows: list[dict[str, Any]] = []
    try:
        for offset in range(args.roots):
            seed = args.seed + offset
            warmup, warmup_index = sample_history_packets(histories, seed=seed)
            context = collect_executed_context(env, seed=seed, history_len=3, action_block=5, warmup_packets=warmup, warmup_policy=WARMUP_POLICY + ".contiguous_history")
            pixels, goal, history = p0._prepare(preprocessor, context["frames"], context["root_frame"], context["goal"], np.asarray(context["history_action_blocks"], dtype=np.float32), device)
            query = p0._latent_context(model, pixels, history)[0, -1].detach().cpu().numpy()
            neighbors = _nearest_source_indices(query, source_embeddings, count=args.proposals - 1)
            bank = torch.cat((torch.zeros((1, 1, history.shape[-1]), device=device), _raw_packets_to_params(source_packets[neighbors], mean=mean, std=std, lower=lower, upper=upper)), dim=0)
            def selected_program(current_pixels, current_goal, current_history, root_seed):
                params, receipt = p1._compile_reconfiguration(model, current_pixels, current_goal, current_history,
                    reconfiguration_horizon=1, future_horizon=2, action_lower=lower, action_upper=upper, damping=args.damping,
                    proposals=args.proposals, non_greedy_margin=1.0e-6, reachability_margin=0.0, seed=root_seed, proposal_bank=bank)
                if params is None:
                    raise RuntimeError("state-conditioned bank produced no eligible P1 prefix")
                return params, receipt
            def zero_program(_pixels, _goal, current_history, _root_seed):
                return torch.zeros((1, 1, current_history.shape[-1]), device=device), {"program_type": "physical_zero_prefix_control"}
            zero = runner._two_stage(name="state_conditioned_zero", first_program=zero_program, env=env, seed=seed, context=context, preprocessor=preprocessor, model=model, mean=mean, std=std, lower=lower, upper=upper, action_block=5, progress_horizon=2, damping=args.damping, device=device)
            try:
                selected = runner._two_stage(name="state_conditioned_reconfiguration", first_program=selected_program, env=env, seed=seed, context=context, preprocessor=preprocessor, model=model, mean=mean, std=std, lower=lower, upper=upper, action_block=5, progress_horizon=2, damping=args.damping, device=device)
            except RuntimeError as error:
                if "no eligible P1 prefix" not in str(error):
                    raise
                rows.append({"seed": seed, "contiguous_warmup_sequence_index": warmup_index, "retrieved_source_rows": source_rows[neighbors].tolist(), "reconfiguration": None, "zero_prefix": zero, "ineligible_reason": str(error)})
                continue
            rows.append({"seed": seed, "contiguous_warmup_sequence_index": warmup_index, "retrieved_source_rows": source_rows[neighbors].tolist(), "reconfiguration": selected, "zero_prefix": zero, "ineligible_reason": None})
    finally:
        env.close()
    eligible = [row for row in rows if row["reconfiguration"] is not None]
    observed = np.asarray([row["reconfiguration"]["observed_first_chunk"]["candidate_goal_cost"] - row["zero_prefix"]["observed_first_chunk"]["candidate_goal_cost"] for row in eligible], dtype=np.float64)
    result = {"schema": "reacher_d2p_state_conditioned_support_audit.v1", "status": "completed", "scope": "fresh CUDA diagnostic only: frozen-latent nearest-neighbor source action retrieval; qpos is evaluator-only and no artifact updates the world model", "inputs": {"policy": str(policy), "weights_sha256": _sha256(policy / "weights.pt"), "adapter_artifact": str(adapter), "adapter_sha256": _sha256(adapter), "official_dataset": str(dataset), "official_dataset_sha256": dataset_sha256, "qpos_readout_binding": binding}, "protocol": {"device": "cuda", "roots": args.roots, "root_seeds": [row["seed"] for row in rows], "source_index_size": args.source_index_size, "source_index_seed": args.source_index_seed, "source_index_rows_sha256": hashlib.sha256(source_rows.tobytes()).hexdigest(), "proposals": args.proposals, "proposal_distribution": "frozen_latent_nearest_source_context_next_packet_with_raw_zero", "warmup_policy": WARMUP_POLICY + ".contiguous_history", "contiguous_warmup_support": history_support}, "aggregate": {"eligible_root_count": len(eligible), "ineligible_root_count": len(rows) - len(eligible), "observed_selected_minus_zero_goal_cost": observed.tolist(), "observed_non_greedy_every_eligible_root": bool(len(eligible) and np.all(observed >= -1.0e-8))}, "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
