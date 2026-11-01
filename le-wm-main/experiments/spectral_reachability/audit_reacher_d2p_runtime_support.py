"""Compare uniform and source-supported Reacher warmup contexts on host CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

import gymnasium as gym
import h5py
import numpy as np
import torch

from experiments.audit_live_reacher_shared_candidate_ranking import _Preprocessor
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from experiments.run_live_reacher_planning import ReacherGoalImageWrapper, _load_action_stats, install_missing_field_compat
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability.preflight_reacher_d2p_p0 import _normalized_action_bounds
from experiments.spectral_reachability.reacher_d2p_context import (
    advance_executed_context,
    capture_execution_snapshot,
    collect_executed_context,
    rgb_digest,
    restore_execution_snapshot,
)
from experiments.spectral_reachability.reacher_d2p_packet_support import WARMUP_POLICY as SUPPORT_WARMUP_POLICY, load_packet_pool, receipt_source, warmup_packets


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _successor_error(
    *, model: torch.nn.Module, preprocessor: _Preprocessor, env: gym.Env, context: dict[str, Any],
    seed: int, mean: np.ndarray, std: np.ndarray, lower: torch.Tensor, upper: torch.Tensor, device: torch.device,
) -> float:
    pixels, goal, history = p0._prepare(preprocessor, context["frames"], context["root_frame"], context["goal"], np.asarray(context["history_action_blocks"], dtype=np.float32), device)
    params = torch.zeros((1, 1, history.shape[-1]), device=device)
    predicted = p0._terminal_batch(model, pixels, goal, history, params, action_lower=lower, action_upper=upper)[0]
    raw, _ = p0._to_raw_actions(params, mean, std, lower, upper, env.action_space.low, env.action_space.high)
    root, goal_rgb = restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"])
    if not np.array_equal(root, context["root_frame"]) or not np.array_equal(goal_rgb, context["goal"]):
        raise RuntimeError("runtime-support audit could not restore its observed RGB root")
    # The previous audit accidentally encoded ``goal`` again here (the second
    # output of ``_prepare``), which measured predicted-successor-to-goal
    # distance rather than model-to-observed-successor fidelity. Advance the
    # real context through exactly the raw-zero packet and encode its rendered
    # successor with the corresponding shifted RGB/action history.
    advanced = advance_executed_context(
        env, context=context, raw_action_block=raw, action_block=raw.shape[0],
    )
    successor_pixels, _successor_goal, successor_history = p0._prepare(
        preprocessor, advanced["frames"], advanced["root_frame"], advanced["goal"],
        np.asarray(advanced["history_action_blocks"], dtype=np.float32), device,
    )
    actual = p0._latent_context(model, successor_pixels, successor_history)[0, -1]
    return float(torch.linalg.vector_norm(predicted - actual).detach())


def _source_state_context(
    env: gym.Env, *, dataset: Path, seed: int, history_len: int = 3, action_block: int = 5,
) -> dict[str, Any]:
    """Rehydrate a held-out official state/history as a live planning root.

    This is a diagnostic context, not a new efficacy protocol. It distinguishes
    action-only support (official packets from an arbitrary deployment reset)
    from the checkpoint's actual source-state/source-history distribution.  The
    qpos-match target is regenerated from that restored source root, rather
    than incorrectly retaining the unrelated reset root's target image.
    """

    env.reset(seed=int(seed))
    with h5py.File(dataset, "r", swmr=True) as handle:
        lengths, offsets = np.asarray(handle["ep_len"], dtype=np.int64), np.asarray(handle["ep_offset"], dtype=np.int64)
        required = (history_len + 1) * action_block
        eligible = np.flatnonzero(lengths >= required)
        if not len(eligible):
            raise ValueError("official source has no state-history window for the declared Reacher context")
        rng = np.random.default_rng(int(seed) ^ 0x5A7E)
        episode = int(eligible[int(rng.integers(len(eligible)))])
        local = int(rng.integers(0, int(lengths[episode]) - required + 1))
        start = int(offsets[episode]) + local
        frame_rows = start + np.arange(history_len + 1, dtype=np.int64) * action_block
        frame_states = np.asarray(handle["qpos"][frame_rows], dtype=np.float64)
        frame_velocities = np.asarray(handle["qvel"][frame_rows], dtype=np.float64)
        packets = np.asarray(handle["action"][start:start + history_len * action_block], dtype=np.float32).reshape(history_len, action_block, -1)
    raw = env.unwrapped
    physics = raw.env.physics
    if frame_states.shape[1:] != physics.data.qpos.shape or frame_velocities.shape[1:] != physics.data.qvel.shape:
        raise ValueError("official source state shape differs from deployed Reacher")
    # Bind the source-root task before *any* history frame is rendered.  The
    # wrapper mutates the scene's cosmetic target geometry while rendering the
    # goal.  Binding it only after the history loop leaves
    # ``[x_t,x_t+5,x_t+10]`` under the reset root's target transaction while
    # ``x_t+15`` and ``goal`` use the source root's transaction.
    physics.data.qpos[:] = frame_states[-1]
    physics.data.qvel[:] = frame_velocities[-1]
    physics.forward()
    goal = env.set_goal_from_state(frame_states[-1], frame_velocities[-1])
    frames = []
    for qpos, qvel in zip(frame_states[:-1], frame_velocities[:-1]):
        physics.data.qpos[:] = qpos
        physics.data.qvel[:] = qvel
        physics.forward()
        frames.append(np.asarray(env.render(), dtype=np.uint8).copy())
    physics.data.qpos[:] = frame_states[-1]
    physics.data.qvel[:] = frame_velocities[-1]
    physics.forward()
    # The source-root goal was bound before the history render transaction;
    # retain that exact target while restoring the observed root snapshot.
    root = np.asarray(env.render(), dtype=np.uint8).copy()
    snapshot = capture_execution_snapshot(env)
    if snapshot is None:
        raise RuntimeError("source-state diagnostic requires a physical execution snapshot")
    raw_steps = packets.reshape(-1, packets.shape[-1])
    return {
        "frames": frames,
        "goal": np.asarray(goal, dtype=np.uint8).copy(),
        "history_action_blocks": [packet.reshape(-1).copy() for packet in packets],
        "warmup_raw_steps": raw_steps,
        "execution_snapshot": snapshot,
        "root_frame": root,
        "context_mode": "official_source_state_history",
        "warmup_policy": "official_source_state_history_v3_goal_bound_before_history_render",
        "source_episode": episode,
        "source_start_row": start,
        "history_rgb_sha256": [rgb_digest(frame) for frame in frames],
        "root_rgb_sha256": rgb_digest(root),
        "goal_rgb_sha256": rgb_digest(np.asarray(goal, dtype=np.uint8)),
        "warmup_actions_sha256": hashlib.sha256(np.ascontiguousarray(raw_steps).tobytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026102510)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("runtime-support audit requires host CUDA exactly; CPU fallback is forbidden")
    if args.roots < 4 or args.output.exists():
        raise ValueError("roots must be at least four and output must be new")
    policy, adapter, output = args.policy.resolve(), args.adapter_artifact.resolve(), args.output.resolve()
    install_missing_field_compat(); device = torch.device("cuda"); torch.empty(0, device=device)
    mean, std = _load_action_stats(adapter, "swm/ReacherDMControl-v0")
    dataset, digest = receipt_source(adapter)
    pool, support = load_packet_pool(dataset, expected_sha256=digest)
    model = load_compatible_pretrained(policy, device=device).eval(); model.requires_grad_(False); model.interpolate_pos_encoding = True
    env = ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65))
    lower_np, upper_np = _normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    lower, upper = torch.as_tensor(lower_np, device=device), torch.as_tensor(upper_np, device=device)
    preprocessor = _Preprocessor(ActionNormalizer(mean, std))
    rows: list[dict[str, Any]] = []
    try:
        for index in range(args.roots):
            seed = int(args.seed + index)
            supported, selected = warmup_packets(pool, history_len=3, seed=seed)
            uniform_context = collect_executed_context(env, seed=seed, history_len=3, action_block=5)
            supported_context = collect_executed_context(env, seed=seed, history_len=3, action_block=5, warmup_packets=supported, warmup_policy=SUPPORT_WARMUP_POLICY)
            source_state_context = _source_state_context(env, dataset=dataset, seed=seed)
            rows.append({"seed": seed, "support_packet_indices": selected.tolist(), "source_state": {key: source_state_context[key] for key in ("source_episode", "source_start_row", "root_rgb_sha256", "warmup_actions_sha256")}, "uniform_successor_latent_l2": _successor_error(model=model, preprocessor=preprocessor, env=env, context=uniform_context, seed=seed, mean=mean, std=std, lower=lower, upper=upper, device=device), "supported_successor_latent_l2": _successor_error(model=model, preprocessor=preprocessor, env=env, context=supported_context, seed=seed, mean=mean, std=std, lower=lower, upper=upper, device=device), "source_state_successor_latent_l2": _successor_error(model=model, preprocessor=preprocessor, env=env, context=source_state_context, seed=seed, mean=mean, std=std, lower=lower, upper=upper, device=device)})
    finally:
        env.close()
    uniform = np.asarray([row["uniform_successor_latent_l2"] for row in rows], dtype=np.float64)
    supported = np.asarray([row["supported_successor_latent_l2"] for row in rows], dtype=np.float64)
    source_state = np.asarray([row["source_state_successor_latent_l2"] for row in rows], dtype=np.float64)
    result: dict[str, Any] = {"schema": "reacher_d2p_runtime_support_audit.v3", "status": "completed", "scope": "fresh CUDA runtime-context diagnostic; no planning root was used for selection", "inputs": {"policy": str(policy), "weights_sha256": _sha256(policy / "weights.pt"), "adapter": str(adapter), "adapter_sha256": _sha256(adapter), "source_packet_support": support}, "protocol": {"device": str(device), "roots": int(args.roots), "root_seeds": [row["seed"] for row in rows], "uniform_warmup_policy": "seeded_full_packet_uniform_v2", "supported_warmup_policy": SUPPORT_WARMUP_POLICY, "source_state_context_policy": "official_source_state_history_v3_goal_bound_before_history_render", "future_probe": "raw-zero five-step packet"}, "aggregate": {"uniform_mean_successor_latent_l2": float(uniform.mean()), "supported_mean_successor_latent_l2": float(supported.mean()), "source_state_mean_successor_latent_l2": float(source_state.mean()), "paired_supported_minus_uniform": (supported - uniform).tolist(), "paired_source_state_minus_supported": (source_state - supported).tolist(), "supported_better_every_root": bool(np.all(supported < uniform)), "source_state_better_every_root_than_supported": bool(np.all(source_state < supported))}, "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
