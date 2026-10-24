"""Build CUDA-only counterfactual pairs from the actual Reacher P1 proposal law.

The first repair source paired a random five-step packet with raw zero.  P1,
however, deploys a packet selected from a physical-uniform bank because it is
locally non-greedy but improves the recompiled bounded-DLS suffix score.  This
builder records that *same* selected-packet-versus-zero contrast at independent
executed RGB roots.  It is source data only: no P0/P1 calibration or outcome
root may occur in this pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import gymnasium as gym

from experiments.audit_live_reacher_shared_candidate_ranking import _Preprocessor
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from experiments.run_live_reacher_planning import ReacherGoalImageWrapper, _load_action_stats, install_missing_field_compat
from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability.preflight_reacher_d2p_p0 import _normalized_action_bounds, _sha256_file
from experiments.spectral_reachability.reacher_d2p_context import advance_executed_context, collect_executed_context, restore_execution_snapshot


SCHEMA = "reacher_d2p_p1_proposal_pair_pool.v1"
HISTORY_LEN = 3
ACTION_BLOCK = 5


def _require_cuda(device_name: str) -> torch.device:
    if device_name != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("P1 proposal pair generation requires host CUDA exactly; CPU fallback is forbidden")
    device = torch.device("cuda")
    if torch.empty(1, device=device).device.type != "cuda":
        raise RuntimeError("CUDA allocation did not land on CUDA")
    return device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _append(store: h5py.Dataset, values: list[np.ndarray]) -> None:
    if not values:
        return
    start = int(store.shape[0])
    data = np.asarray(values, dtype=store.dtype)
    store.resize((start + len(data), *store.shape[1:]))
    store[start:] = data


def build(
    *, output: Path, policy: Path, adapter_artifact: Path, seed: int, roots: int,
    device_name: str, damping: float, proposals: int, forbidden_seeds: set[int],
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable P1 proposal pool: {output}")
    if roots < 2 or proposals < 2 or damping <= 0.0:
        raise ValueError("P1 proposal pool dimensions are invalid")
    seeds = [int(seed + index) for index in range(roots)]
    overlap = sorted(set(seeds) & {int(value) for value in forbidden_seeds})
    if overlap:
        raise ValueError(f"P1 proposal training roots overlap forbidden roots: {overlap[:8]}")
    device = _require_cuda(device_name)
    install_missing_field_compat()
    policy, adapter_artifact = policy.resolve(), adapter_artifact.resolve()
    mean, std = _load_action_stats(adapter_artifact, "swm/ReacherDMControl-v0")
    model = load_compatible_pretrained(policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    preprocessor = _Preprocessor(ActionNormalizer(mean, std))
    env = ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"))
    action_dim = int(np.prod(env.action_space.shape))
    p0._validate_reacher_action_protocol(model, mean, std, action_dim=action_dim, action_block=ACTION_BLOCK)
    low, high = _normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    lower, upper = torch.as_tensor(low, device=device), torch.as_tensor(high, device=device)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    pairs_path = output / "pairs.h5"
    selected = 0
    skipped: list[int] = []
    try:
        with h5py.File(pairs_path, "x") as handle:
            image = dict(shape=(0, 224, 224, 3), maxshape=(None, 224, 224, 3), chunks=(4, 224, 224, 3), compression="gzip", compression_opts=1, dtype=np.uint8)
            context_store = handle.create_dataset("context_frames", shape=(0, HISTORY_LEN, 224, 224, 3), maxshape=(None, HISTORY_LEN, 224, 224, 3), chunks=(1, HISTORY_LEN, 224, 224, 3), compression="gzip", compression_opts=1, dtype=np.uint8)
            root_store = handle.create_dataset("root_frame", **image)
            goal_store = handle.create_dataset("goal_frame", **image)
            candidate_store = handle.create_dataset("candidate_successor", **image)
            zero_store = handle.create_dataset("zero_successor", **image)
            history_store = handle.create_dataset("history_raw_actions", shape=(0, HISTORY_LEN, ACTION_BLOCK * action_dim), maxshape=(None, HISTORY_LEN, ACTION_BLOCK * action_dim), chunks=(64, HISTORY_LEN, ACTION_BLOCK * action_dim), compression="gzip", compression_opts=1, dtype=np.float32)
            action_store = handle.create_dataset("candidate_raw_action", shape=(0, ACTION_BLOCK * action_dim), maxshape=(None, ACTION_BLOCK * action_dim), chunks=(64, ACTION_BLOCK * action_dim), compression="gzip", compression_opts=1, dtype=np.float32)
            episode_store = handle.create_dataset("episode_index", shape=(0,), maxshape=(None,), chunks=(256,), dtype=np.int64)
            seed_store = handle.create_dataset("source_seed", shape=(0,), maxshape=(None,), chunks=(256,), dtype=np.int64)
            proposal_store = handle.create_dataset("selected_proposal_index", shape=(0,), maxshape=(None,), chunks=(256,), dtype=np.int64)
            for episode_index, root_seed in enumerate(seeds):
                context = collect_executed_context(env, seed=root_seed, history_len=HISTORY_LEN, action_block=ACTION_BLOCK)
                pixels, goal, history = p0._prepare(
                    preprocessor, context["frames"], context["root_frame"], context["goal"],
                    np.asarray(context["history_action_blocks"], dtype=np.float32), device,
                )
                params, receipt = p1._compile_reconfiguration(
                    model, pixels, goal, history, reconfiguration_horizon=1, future_horizon=2,
                    action_lower=lower, action_upper=upper, damping=damping, proposals=proposals,
                    non_greedy_margin=1.0e-6, reachability_margin=1.0e-6, seed=root_seed,
                    proposal_distribution="physical_uniform", selection_metric="bounded_dls_suffix_goal_cost",
                )
                if params is None:
                    skipped.append(root_seed)
                    continue
                raw, _ = p0._to_raw_actions(params, mean, std, lower, upper, env.action_space.low, env.action_space.high)
                raw = np.asarray(raw, dtype=np.float32)
                restore_execution_snapshot(env, seed=root_seed, snapshot=context["execution_snapshot"])
                candidate = advance_executed_context(env, context=context, raw_action_block=raw, action_block=ACTION_BLOCK)
                restore_execution_snapshot(env, seed=root_seed, snapshot=context["execution_snapshot"])
                zero = advance_executed_context(env, context=context, raw_action_block=np.zeros_like(raw), action_block=ACTION_BLOCK)
                _append(context_store, [np.stack(context["frames"])])
                _append(root_store, [context["root_frame"]]); _append(goal_store, [context["goal"]])
                _append(candidate_store, [candidate["root_frame"]]); _append(zero_store, [zero["root_frame"]])
                _append(history_store, [np.asarray(context["history_action_blocks"], dtype=np.float32)])
                _append(action_store, [raw.reshape(-1)])
                _append(episode_store, [np.asarray(episode_index, dtype=np.int64)])
                _append(seed_store, [np.asarray(root_seed, dtype=np.int64)])
                _append(proposal_store, [np.asarray(receipt["selected_proposal_index"], dtype=np.int64)])
                selected += 1
    finally:
        env.close()
    if selected < 2:
        raise RuntimeError("P1 proposal pool has fewer than two model-selected packets")
    result = {
        "schema": SCHEMA, "status": "completed", "pairs_hdf5": str(pairs_path.resolve()),
        "pairs_hdf5_sha256": _sha256(pairs_path), "environment": "swm/ReacherDMControl-v0",
        "cuda_host": {"name": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda},
        "seed_range": [seeds[0], seeds[-1]], "roots_requested": roots, "selected_pairs": selected,
        "ineligible_source_seeds": skipped, "p0_p1_calibration_or_outcome_roots_used": False,
        "proposal_contract": {"policy": str(policy), "weights_sha256": _sha256_file(policy / "weights.pt"), "adapter_sha256": _sha256_file(adapter_artifact), "history_len": HISTORY_LEN, "action_block": ACTION_BLOCK, "damping": damping, "proposals": proposals, "proposal_distribution": "physical_uniform", "selection_metric": "bounded_dls_suffix_goal_cost", "selection": "same P1 bounded-DLS non-greedy selector against raw-zero packet"},
        "claim_boundary": "Independent P1 proposal-distribution training source only; fresh disjoint CUDA P0/P1 gates remain mandatory.",
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True); parser.add_argument("--seed", type=int, default=20268100)
    parser.add_argument("--roots", type=int, default=256); parser.add_argument("--damping", type=float, default=10.0)
    parser.add_argument("--proposals", type=int, default=128); parser.add_argument("--forbidden-seed", type=int, action="append", default=[])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = build(output=args.output, policy=args.policy, adapter_artifact=args.adapter_artifact, seed=args.seed, roots=args.roots, device_name=args.device, damping=args.damping, proposals=args.proposals, forbidden_seeds=set(args.forbidden_seed))
    print(json.dumps({"status": result["status"], "selected_pairs": result["selected_pairs"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
