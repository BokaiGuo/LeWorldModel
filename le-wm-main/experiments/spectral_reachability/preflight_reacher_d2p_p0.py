"""CUDA-only D2P-P0 preflight for the frozen RGB Reacher LeWM.

This audit exists because :meth:`LeWM.rollout` consumes the first ``H`` rows
of its candidate action tensor as the action history.  It does *not* consume
``info['action']`` when rolling out.  Passing a future-only CEM candidate
therefore lets optimized future actions silently replace the observed action
history.  That is an implementation/contract defect, not a negative result
about planning.

The adapter below binds an explicit, recorded normalized action history to the
candidate prefix and differentiates only a bounded future suffix.  It checks
the action layout, frozen parameter hash, non-zero finite Jacobian and central
finite-difference agreement before any D2P comparison is permitted.  It never
falls back to CPU and does not use simulator qpos/qvel as a model input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

# Keep CUDA initialization ahead of creating a MuJoCo environment.  The
# fail-closed availability check itself belongs in ``main`` so helper-level
# unit tests can import this module without being mistaken for an experiment.
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import gymnasium as gym

from experiments.audit_live_reacher_shared_candidate_ranking import _Preprocessor
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.online_planning_protocol import validate_action_protocol
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from experiments.run_live_reacher_planning import (
    ReacherGoalImageWrapper,
    _load_action_stats,
    install_missing_field_compat,
)
from experiments.spectral_reachability.reacher_d2p_context import (
    WARMUP_POLICY,
    collect_executed_context,
    require_new_output,
)
from experiments.spectral_reachability.build_reacher_d2p_training_action_stats import validate_training_semantic_receipt


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parameter_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def _normalized_block(
    raw_action: np.ndarray, mean: np.ndarray, std: np.ndarray, action_block: int,
) -> np.ndarray:
    raw_action = np.asarray(raw_action, dtype=np.float32).reshape(1, -1)
    block = np.repeat(raw_action, int(action_block), axis=0).reshape(-1)
    if block.shape != mean.shape or std.shape != mean.shape:
        raise ValueError("Reacher action normalization dimensions are incompatible")
    return (block - mean) / np.maximum(std, 1.0e-6)


def _raw_block(raw_action: np.ndarray, action_block: int) -> np.ndarray:
    """Repeat a raw environment action into one model action block.

    ``_Preprocessor._prepare_info`` owns action normalization.  Returning raw
    values here prevents the observed history prefix from being normalized a
    second time while the optimized future suffix stays single-normalized.
    """
    raw_action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    if raw_action.size < 1 or int(action_block) < 1:
        raise ValueError("raw action and action_block must be non-empty")
    return np.tile(raw_action[None, :], (int(action_block), 1)).reshape(-1)


def _normalized_action_bounds(
    mean: np.ndarray, std: np.ndarray, raw_low: np.ndarray, raw_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map the environment's raw action box into checkpoint coordinates.

    A normalized symmetric box (for example ``[-3, 3]``) is not generally a
    physical action box after de-normalization.  Planning outside this returned
    interval would make MuJoCo clip execution while the world model scores an
    unattainable sequence.
    """

    mean, std = np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
    raw_low, raw_high = np.asarray(raw_low, dtype=np.float32).reshape(-1), np.asarray(raw_high, dtype=np.float32).reshape(-1)
    if mean.ndim != 1 or std.shape != mean.shape or mean.size % raw_low.size or raw_low.shape != raw_high.shape or not np.all(raw_low < raw_high):
        raise ValueError("incompatible normalized-action and environment-action bounds")
    repeat = mean.size // raw_low.size
    lower = (np.tile(raw_low, repeat) - mean) / np.maximum(std, 1.0e-6)
    upper = (np.tile(raw_high, repeat) - mean) / np.maximum(std, 1.0e-6)
    if not np.all(lower < upper):
        raise ValueError("normalized action bounds are degenerate")
    return lower, upper


def _bounded_normalized_actions(params: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Smoothly parameterize exactly the physical environment action box."""

    if lower.ndim != 1 or upper.shape != lower.shape or params.shape[-1] != lower.numel() or not torch.all(lower < upper):
        raise ValueError("future-action parameters and normalized bounds are incompatible")
    midpoint, radius = (lower + upper) / 2.0, (upper - lower) / 2.0
    return midpoint + radius * torch.tanh(params)


def _prepare_context(
    preprocessor: _Preprocessor,
    frames: list[np.ndarray],
    root_frame: np.ndarray,
    history_actions: list[np.ndarray],
    goal: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(frames) != len(history_actions) or len(frames) < 2:
        raise ValueError("RGB/action history lengths differ")
    anchored_frames = [np.asarray(frame, dtype=np.uint8) for frame in frames[1:]]
    anchored_frames.append(np.asarray(root_frame, dtype=np.uint8))
    raw = {
        "pixels": np.asarray(anchored_frames, dtype=np.uint8)[None],
        "goal": np.asarray(goal, dtype=np.uint8)[None, None],
        "action": np.asarray(history_actions, dtype=np.float32)[None],
    }
    prepared = preprocessor._prepare_info(raw)
    return (
        prepared["pixels"].to(device),
        prepared["goal"].to(device),
        prepared["action"].to(device),
    )


def _frozen_context_embedding(
    model: torch.nn.Module, pixels: torch.Tensor, actions: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        encoded = model.encode({"pixels": pixels, "action": actions})["emb"]
    return encoded.detach()


def _terminal_latent(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    goal: torch.Tensor,
    history_actions: torch.Tensor,
    future_actions: torch.Tensor,
    *,
    history_len: int,
) -> torch.Tensor:
    """Return a terminal prediction anchored at the observed post-warmup root."""
    if future_actions.ndim != 2:
        raise ValueError("future actions must have shape [future_horizon, action_dim]")
    if tuple(history_actions.shape) != (1, history_len, future_actions.shape[-1]):
        raise ValueError("history action prefix violates the declared D2P layout")
    if pixels.shape[1] != history_len or history_len < 2:
        raise ValueError("root-anchored pixel history violates the declared D2P layout")
    context_emb = _frozen_context_embedding(model, pixels, history_actions)
    action_context = torch.cat([history_actions[:, 1:], future_actions[:1][None]], dim=1)
    emb_context = context_emb
    terminal = None
    for step in range(future_actions.shape[0]):
        action_emb = model.action_encoder(action_context[:, -history_len:])
        terminal = model.predict(emb_context[:, -history_len:], action_emb)[:, -1]
        if step + 1 < future_actions.shape[0]:
            emb_context = torch.cat([emb_context, terminal[:, None]], dim=1)
            action_context = torch.cat([action_context, future_actions[step + 1:step + 2][None]], dim=1)
    if terminal is None or terminal.ndim != 2 or terminal.shape[0] != 1:
        raise RuntimeError("unexpected root-anchored predicted embedding shape")
    return terminal[0]


def _jacobian_and_fd(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    goal: torch.Tensor,
    history_actions: torch.Tensor,
    *,
    history_len: int,
    future_horizon: int,
    probe_dims: int,
    fd_pairs: int,
    epsilon: float,
    action_lower: torch.Tensor,
    action_upper: torch.Tensor,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    action_dim = int(history_actions.shape[-1])
    latent = _terminal_latent(
        model, pixels, goal, history_actions,
        _bounded_normalized_actions(torch.zeros((future_horizon, action_dim), device=device), action_lower, action_upper),
        history_len=history_len,
    )
    latent_dim = int(latent.numel())
    generator = torch.Generator(device=device).manual_seed(int(seed))
    projection = torch.randn((probe_dims, latent_dim), generator=generator, device=device)
    projection = projection / projection.norm(dim=1, keepdim=True).clamp_min(1.0e-8)

    def projected(flat_future: torch.Tensor) -> torch.Tensor:
        future = _bounded_normalized_actions(
            flat_future.reshape(future_horizon, action_dim), action_lower, action_upper,
        )
        return projection @ _terminal_latent(
            model, pixels, goal, history_actions, future, history_len=history_len,
        )

    origin = torch.zeros(future_horizon * action_dim, device=device, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(projected, origin, vectorize=True)
    if not torch.isfinite(jacobian).all() or float(jacobian.norm()) <= 1.0e-10:
        raise RuntimeError("future-suffix action Jacobian is non-finite or zero")
    pairs = min(int(fd_pairs), int(probe_dims), int(origin.numel()))
    fd_rows: list[dict[str, float | int]] = []
    for pair in range(pairs):
        output_index = pair % int(probe_dims)
        input_index = (3 * pair + 1) % int(origin.numel())
        plus = origin.detach().clone()
        minus = origin.detach().clone()
        plus[input_index] += float(epsilon)
        minus[input_index] -= float(epsilon)
        with torch.no_grad():
            finite = float((projected(plus)[output_index] - projected(minus)[output_index]) / (2.0 * epsilon))
        autograd = float(jacobian[output_index, input_index])
        absolute = abs(autograd - finite)
        relative = absolute / max(abs(autograd), abs(finite), 1.0e-6)
        fd_rows.append({
            "output_index": output_index,
            "future_action_index": input_index,
            "autograd": autograd,
            "central_fd": finite,
            "absolute_error": absolute,
            "relative_error": relative,
        })
    singular = torch.linalg.svdvals(jacobian)
    return {
        "action_layout": {
            "history_prefix_rows": history_len,
            "optimized_future_rows": future_horizon,
            "action_dim": action_dim,
            "jacobian_columns": int(jacobian.shape[1]),
            "future_only": True,
        },
        "jacobian_shape": list(jacobian.shape),
        "frobenius_norm": float(jacobian.norm()),
        "singular_values": [float(value) for value in singular],
        "effective_rank_1e-4": int((singular > singular[0] * 1.0e-4).sum()),
        "finite_difference": fd_rows,
        "median_relative_error": float(np.median([row["relative_error"] for row in fd_rows])),
        "max_relative_error": float(max(row["relative_error"] for row in fd_rows)),
    }


def _collect_root(
    env: gym.Env,
    seed: int,
    *,
    history_len: int,
    action_block: int,
) -> dict[str, Any]:
    return collect_executed_context(
        env, seed=int(seed), history_len=int(history_len), action_block=int(action_block),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=Path("/home/gbk/.stable_worldmodel/checkpoints/reacher/lewm"))
    parser.add_argument("--adapter-artifact", type=Path, default=Path("results/live_cross_environment_prediction_20261015/reacher_cuda/cross_environment_prediction.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--roots", type=int, default=4)
    parser.add_argument("--history-len", type=int, default=3)
    parser.add_argument("--future-horizon", type=int, default=2)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--probe-dims", type=int, default=8)
    parser.add_argument("--fd-pairs", type=int, default=8)
    parser.add_argument("--fd-epsilon", type=float, default=1.0e-3)
    parser.add_argument("--target-delta", type=float, nargs=2, default=(0.65, -0.65))
    args = parser.parse_args()
    require_new_output(args.output, receipt="P0 preflight output")
    if args.device != "cuda":
        raise RuntimeError("D2P-P0 requires --device cuda exactly; no CPU substitute")
    if not torch.cuda.is_available():
        raise RuntimeError("Reacher D2P-P0 is CUDA-only; CPU fallback is forbidden")
    if args.roots < 2 or args.history_len < 1 or args.future_horizon < 1:
        raise ValueError("roots >= 2 and positive history/future horizons are required")
    if args.fd_epsilon <= 0:
        raise ValueError("finite-difference epsilon must be positive")

    install_missing_field_compat()
    device = torch.device("cuda")
    validate_training_semantic_receipt(args.adapter_artifact)
    mean, std = _load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    model = load_compatible_pretrained(args.policy, device=args.device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    parameter_before = _parameter_hash(model)
    preprocessor = _Preprocessor(ActionNormalizer(mean, std))
    env = ReacherGoalImageWrapper(
        gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"),
        target_delta=args.target_delta,
    )
    action_dim = int(np.prod(env.action_space.shape))
    checkpoint_action_block = validate_action_protocol(
        model, action_dim=action_dim, action_block=args.action_block,
    )
    if mean.shape != (checkpoint_action_block * action_dim,) or std.shape != mean.shape:
        raise ValueError(
            "Reacher action statistics do not match the checkpoint action protocol: "
            f"expected flattened width {checkpoint_action_block * action_dim}, "
            f"got mean={mean.shape}, std={std.shape}"
        )
    normalized_low_np, normalized_high_np = _normalized_action_bounds(
        mean, std, env.action_space.low, env.action_space.high,
    )
    action_lower = torch.as_tensor(normalized_low_np, device=device)
    action_upper = torch.as_tensor(normalized_high_np, device=device)
    roots: list[dict[str, Any]] = []
    started = time.time()
    try:
        for index in range(args.roots):
            seed = int(args.seed + index)
            context = _collect_root(
                env, seed, history_len=args.history_len, action_block=args.action_block,
            )
            pixels, prepared_goal, prepared_history = _prepare_context(
                preprocessor, context["frames"], context["root_frame"],
                context["history_action_blocks"], context["goal"], device,
            )
            audit = _jacobian_and_fd(
                model, pixels, prepared_goal, prepared_history,
                history_len=args.history_len,
                future_horizon=args.future_horizon,
                probe_dims=args.probe_dims,
                fd_pairs=args.fd_pairs,
                epsilon=args.fd_epsilon,
                action_lower=action_lower,
                action_upper=action_upper,
                seed=seed,
                device=device,
            )
            roots.append({
                "seed": seed,
                "jacobian": audit,
                "executed_context": {key: context[key] for key in (
                    "context_mode", "warmup_policy", "warmup_steps",
                    "history_rgb_sha256", "root_rgb_sha256", "goal_rgb_sha256",
                    "warmup_actions_sha256",
                )},
            })
            print(json.dumps({"seed": seed, "jacobian_norm": audit["frobenius_norm"], "fd_median_relative_error": audit["median_relative_error"]}, sort_keys=True))
    finally:
        env.close()
    parameter_after = _parameter_hash(model)
    norms = [row["jacobian"]["frobenius_norm"] for row in roots]
    fd_errors = [row["jacobian"]["median_relative_error"] for row in roots]
    gates = {
        "cuda_only": True,
        "frozen_parameter_hash_unchanged": parameter_before == parameter_after,
        "every_root_nonzero_future_suffix_jacobian": bool(min(norms) > 1.0e-10),
        "central_fd_median_relative_error_le_0_10_each_root": bool(max(fd_errors) <= 0.10),
        "candidate_prefix_matches_declared_history": True,
        "observed_post_warmup_root_anchors_model_rollout": True,
        "checkpoint_action_block_matches_requested": True,
    }
    checkpoint_weight = args.policy / "weights.pt"
    report = {
        "schema": "d2p_reacher_p0_preflight.v1",
        "status": "pass" if all(gates.values()) else "fail",
        "scope": "CUDA-only frozen Reacher D2P future-suffix Jacobian admission; no physical planning conclusion",
        "implementation_finding": {
            "upstream_rollout_contract": "LeWM.rollout reads candidate[:history_len] as action history and does not consume info['action']",
            "risk_to_prior_generic_cem": "a future-only candidate may overwrite the observed-action prefix unless an adapter binds it explicitly",
            "this_audit": "uses the observed post-warmup RGB root as the final context frame, binds its two preceding causal action tokens, and differentiates only future actions",
        },
        "cuda_host_preflight": {
            "device": str(device),
            "name": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
        "inputs": {
            "checkpoint": str(args.policy.resolve()),
            "weights_sha256": _sha256_file(checkpoint_weight),
            "adapter_artifact": str(args.adapter_artifact.resolve()),
            "adapter_sha256": _sha256_file(args.adapter_artifact),
            "history_context": "causally executed RGB/action warmup; reset padding is forbidden",
            "target_delta": [float(value) for value in args.target_delta],
            "environment_action_box_raw": [np.asarray(env.action_space.low).tolist(), np.asarray(env.action_space.high).tolist()],
            "model_action_box_normalized": [normalized_low_np.tolist(), normalized_high_np.tolist()],
        },
        "protocol": {
            "runtime_input": "causally executed rendered RGB/action history and rendered goal RGB",
            "simulator_state_used_as_model_input": False,
            "history_len": args.history_len,
            "future_horizon": args.future_horizon,
            "action_block": args.action_block,
            "checkpoint_action_block": checkpoint_action_block,
            "action_layout": "flattened time-major [action_block, action_dim]",
            "action_parameterization": "tanh mapped to the exact environment box after raw-to-checkpoint normalization",
            "context_mode": "executed_warmup",
            "warmup_policy": WARMUP_POLICY,
            "warmup_steps": int(args.history_len * args.action_block),
            "rollout_root_conditioning": "observed_post_warmup_rgb_anchor",
            "roots": args.roots,
            "root_seeds": [row["seed"] for row in roots],
        },
        "parameter_hash_before": parameter_before,
        "parameter_hash_after": parameter_after,
        "gates": gates,
        "aggregate": {
            "jacobian_frobenius_norms": norms,
            "fd_median_relative_errors": fd_errors,
        },
        "roots": roots,
        "elapsed_seconds": time.time() - started,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "gates": gates, "output": str(args.output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
