"""CUDA-only implementation admission for the independent TwoRoom D2P path.

This is deliberately a P0 receipt, not a planning-efficacy result.  It closes
the two input-contract gaps that invalidate older TwoRoom diagnostics: action
statistics must be the two-dimensional training statistics tiled into a
time-major five-action packet, and the rollout prefix must be actual actions
executed before the planning boundary.  The frozen model sees rendered RGB
history, a rendered goal image, and that causal packet history only; evaluator
state never crosses into the model path.

The checkpoint rollout treats its first ``history_len`` action tokens as the
observed prefix.  Future controls are therefore differentiated only after that
prefix, while the observed post-prefix RGB root is retained as a fidelity
anchor rather than silently substituted with padding or an adapter artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401 -- registers swm/TwoRoom-v1
import torch

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.gpu_preflight import require_cuda_allocation
from experiments.online_planning_protocol import validate_action_protocol
from experiments.run_live_tworoom_planning import TwoRoomGoalAliasWrapper, _image_transform
from experiments.spectral_reachability.build_tworoom_d2p_training_action_stats import (
    validate_training_semantic_receipt,
)
from experiments.spectral_reachability import tworoom_d2p_packet_support as packet_support


SCHEMA = "tworoom_d2p_p0_preflight.v1"
WARMUP_POLICY = packet_support.WARMUP_POLICY


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _parameter_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def _validate_tworoom_action_protocol(
    model: torch.nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    action_dim: int,
    action_block: int,
) -> int:
    """Bind the released encoder width and raw training statistics together."""

    checkpoint_block = validate_action_protocol(
        model, action_dim=action_dim, action_block=action_block,
    )
    expected_width = int(checkpoint_block * action_dim)
    if mean.shape != (expected_width,) or std.shape != mean.shape:
        raise ValueError(
            "TwoRoom training action statistics do not match the frozen "
            f"checkpoint packet width {expected_width}: mean={mean.shape}, std={std.shape}"
        )
    if not np.array_equal(mean.reshape(action_block, action_dim), np.broadcast_to(mean[:action_dim], (action_block, action_dim))):
        raise ValueError("TwoRoom action means are position-specific rather than training-semantic")
    if not np.array_equal(std.reshape(action_block, action_dim), np.broadcast_to(std[:action_dim], (action_block, action_dim))):
        raise ValueError("TwoRoom action standard deviations are position-specific rather than training-semantic")
    return checkpoint_block


def _normalize_raw_packets(
    raw_packets: torch.Tensor | np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    action_block: int,
    action_dim: int,
) -> torch.Tensor:
    """Z-score actual ``[packet, step, coordinate]`` controls time-major.

    The operation remains in torch for future packets, so the Jacobian reaches
    the action encoder.  It never repeats one action to manufacture a packet:
    every physical step remains independently represented.
    """

    values = raw_packets if torch.is_tensor(raw_packets) else torch.as_tensor(raw_packets, dtype=torch.float32)
    if values.ndim != 3 or tuple(values.shape[-2:]) != (int(action_block), int(action_dim)):
        raise ValueError(
            "TwoRoom raw packets must have shape "
            f"[packets, {action_block}, {action_dim}], got {tuple(values.shape)}"
        )
    if not torch.isfinite(values).all():
        raise ValueError("TwoRoom raw action packets must be finite")
    packet_mean = torch.as_tensor(mean, dtype=values.dtype, device=values.device).reshape(action_block, action_dim)
    packet_std = torch.as_tensor(std, dtype=values.dtype, device=values.device).reshape(action_block, action_dim)
    if not torch.isfinite(packet_mean).all() or not torch.isfinite(packet_std).all() or torch.any(packet_std <= 0.0):
        raise ValueError("TwoRoom action-stat receipt is not finite and positive")
    return ((values - packet_mean) / packet_std).reshape(values.shape[0], -1)


def _bounded_live_actions(params: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Map unconstrained compiler coordinates exactly into the environment box."""

    if params.ndim != 3 or params.shape[-1] != low.numel() or low.shape != high.shape or not torch.all(low < high):
        raise ValueError("TwoRoom future parameters and environment action bounds are incompatible")
    return (low + high) / 2.0 + (high - low) / 2.0 * torch.tanh(params)


def _validated_raw_live_actions(raw_packets: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Pass through an already physical packet after enforcing its action box.

    P0 owns unconstrained compiler coordinates and intentionally applies
    ``tanh``.  A sealed P1 bank instead holds the exact physical packets that
    will be replayed.  Mapping that bank through ``tanh`` would score a
    different candidate than the one sent to MuJoCo.
    """

    if raw_packets.ndim != 3 or raw_packets.shape[-1] != low.numel() or low.shape != high.shape or not torch.all(low < high):
        raise ValueError("TwoRoom raw future packets and environment action bounds are incompatible")
    if not torch.isfinite(raw_packets).all():
        raise ValueError("TwoRoom raw future packets must be finite")
    lower, upper = low.reshape(1, 1, -1), high.reshape(1, 1, -1)
    if torch.any(raw_packets < lower) or torch.any(raw_packets > upper):
        raise ValueError("TwoRoom raw future packet leaves the environment action box")
    return raw_packets


def _image_tensor(image: np.ndarray, *, transform: Any, device: torch.device) -> torch.Tensor:
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("expected one uint8 HWC TwoRoom RGB frame")
    return transform(torch.from_numpy(image).permute(2, 0, 1)).to(device)[None, None]


def _frames_tensor(frames: list[np.ndarray], *, transform: Any, device: torch.device) -> torch.Tensor:
    if not frames:
        raise ValueError("at least one executed TwoRoom RGB history frame is required")
    rows = [_image_tensor(frame, transform=transform, device=device)[0, 0] for frame in frames]
    return torch.stack(rows)[None]


def _root_anchored_action_sequence(history_tokens: torch.Tensor, future_tokens: torch.Tensor) -> torch.Tensor:
    """Shift causal action tokens when the latest RGB frame is observed.

    For observed frames ``[x_{t-2}, x_{t-1}, x_t]``, the action encoder must
    receive ``[a_{t-2}, a_{t-1}, a_t]`` to predict ``x_{t+1}``.  Passing all
    three actions ending at ``a_{t-1}`` instead makes ``rollout`` predict the
    already observed root a second time before touching a candidate.  This
    helper makes the shift explicit and keeps the first candidate packet
    differentiable.
    """

    if history_tokens.ndim != 2 or future_tokens.ndim != 2:
        raise ValueError("root-anchored action tokens must be two-dimensional")
    if history_tokens.shape[0] < 2 or future_tokens.shape[0] < 1 or history_tokens.shape[1] != future_tokens.shape[1]:
        raise ValueError("root-anchored action tokens have incompatible causal dimensions")
    return torch.cat((history_tokens[1:], future_tokens), dim=0)


def _goal_latent(model: torch.nn.Module, goal_rgb: np.ndarray, *, transform: Any, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        encoded = model.encode({"pixels": _image_tensor(goal_rgb, transform=transform, device=device)})["emb"][0, 0]
    if encoded.ndim != 1 or not torch.isfinite(encoded).all():
        raise RuntimeError("frozen TwoRoom model produced an invalid goal embedding")
    return encoded.detach()


def _terminal_prediction(
    model: torch.nn.Module,
    frames: list[np.ndarray],
    root_frame: np.ndarray,
    history_raw_packets: np.ndarray,
    future_params: torch.Tensor,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    action_block: int,
    action_dim: int,
    transform: Any,
    device: torch.device,
    low: torch.Tensor,
    high: torch.Tensor,
    future_packets_are_raw: bool = False,
) -> torch.Tensor:
    """Roll out after a real causal prefix without differentiating the encoder.

    ``future_packets_are_raw`` is only for sealed physical P1 packets.  The
    default remains P0's differentiable unconstrained coordinate interface.
    """

    if len(frames) < 2:
        raise ValueError("TwoRoom root-anchored rollout needs at least two preceding RGB frames")
    pixels = _frames_tensor([*frames[1:], np.asarray(root_frame, dtype=np.uint8)], transform=transform, device=device)
    history = torch.as_tensor(history_raw_packets, dtype=torch.float32, device=device)
    if pixels.shape[1] != history.shape[0]:
        raise ValueError("TwoRoom RGB/action history lengths differ")
    future_raw = (
        _validated_raw_live_actions(future_params, low, high)
        if future_packets_are_raw
        else _bounded_live_actions(future_params, low, high)
    )
    history_tokens = _normalize_raw_packets(
        history, mean, std, action_block=action_block, action_dim=action_dim,
    )
    future_tokens = _normalize_raw_packets(
        future_raw, mean, std, action_block=action_block, action_dim=action_dim,
    )
    actions = _root_anchored_action_sequence(history_tokens, future_tokens)[None, None]
    output = model.rollout({"pixels": pixels[:, None]}, actions, history_size=int(pixels.shape[1]))
    terminal = output["predicted_emb"][0, 0, -1]
    if terminal.ndim != 1 or not torch.isfinite(terminal).all():
        raise RuntimeError("frozen TwoRoom model produced an invalid terminal embedding")
    return terminal


def _jacobian_and_fd(
    model: torch.nn.Module,
    frames: list[np.ndarray],
    root_frame: np.ndarray,
    history_raw_packets: np.ndarray,
    goal_rgb: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    action_block: int,
    action_dim: int,
    future_horizon: int,
    probe_dims: int,
    fd_pairs: int,
    epsilon: float,
    seed: int,
    transform: Any,
    device: torch.device,
    low: torch.Tensor,
    high: torch.Tensor,
) -> dict[str, Any]:
    goal = _goal_latent(model, goal_rgb, transform=transform, device=device)
    origin = torch.zeros(
        (future_horizon, action_block, action_dim),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    with torch.no_grad():
        latent_dim = int(_terminal_prediction(
            model, frames, root_frame, history_raw_packets, origin.detach(), mean=mean, std=std,
            action_block=action_block, action_dim=action_dim, transform=transform,
            device=device, low=low, high=high,
        ).numel())
    if min(probe_dims, fd_pairs, latent_dim) < 1 or epsilon <= 0.0:
        raise ValueError("invalid TwoRoom Jacobian or finite-difference configuration")
    generator = torch.Generator(device=device).manual_seed(int(seed))
    projection = torch.randn((min(int(probe_dims), latent_dim), latent_dim), generator=generator, device=device)
    projection = projection / projection.norm(dim=1, keepdim=True).clamp_min(1.0e-8)

    def projected(flat: torch.Tensor) -> torch.Tensor:
        terminal = _terminal_prediction(
            model, frames, root_frame, history_raw_packets,
            flat.reshape(future_horizon, action_block, action_dim),
            mean=mean, std=std, action_block=action_block, action_dim=action_dim,
            transform=transform, device=device, low=low, high=high,
        )
        return projection @ (goal - terminal)

    flat_origin = origin.reshape(-1)
    jacobian = torch.autograd.functional.jacobian(projected, flat_origin, vectorize=True)
    if jacobian.ndim != 2 or not torch.isfinite(jacobian).all() or float(jacobian.norm()) <= 1.0e-10:
        raise RuntimeError("TwoRoom future-action Jacobian is non-finite or zero")
    fd_rows: list[dict[str, float | int]] = []
    for pair in range(min(int(fd_pairs), int(jacobian.shape[0]), int(jacobian.shape[1]))):
        output_index = pair % int(jacobian.shape[0])
        input_index = (3 * pair + 1) % int(jacobian.shape[1])
        plus, minus = flat_origin.detach().clone(), flat_origin.detach().clone()
        plus[input_index] += float(epsilon)
        minus[input_index] -= float(epsilon)
        with torch.no_grad():
            central = float((projected(plus)[output_index] - projected(minus)[output_index]) / (2.0 * epsilon))
        autograd = float(jacobian[output_index, input_index])
        absolute = abs(autograd - central)
        fd_rows.append({
            "output_index": output_index,
            "future_raw_action_index": input_index,
            "autograd": autograd,
            "central_fd": central,
            "absolute_error": absolute,
            "relative_error": absolute / max(abs(autograd), abs(central), 1.0e-6),
        })
    singular = torch.linalg.svdvals(jacobian)
    return {
        "action_layout": {
            "recorded_causal_history_packets": int(history_raw_packets.shape[0]),
            "root_anchored_input_frames": int(history_raw_packets.shape[0]),
            "root_anchored_executed_action_packets": int(history_raw_packets.shape[0] - 1),
            "first_future_packet_fills_final_action_context_slot": True,
            "future_packets": int(future_horizon),
            "physical_steps_per_packet": int(action_block),
            "raw_action_dim": int(action_dim),
            "flattened_packet_layout": "time-major [a0_x,a0_y,...,a4_x,a4_y]",
            "jacobian_columns": int(jacobian.shape[1]),
            "future_only": True,
        },
        "jacobian_shape": [int(value) for value in jacobian.shape],
        "frobenius_norm": float(jacobian.norm()),
        "singular_values": [float(value) for value in singular],
        "effective_rank_1e-4": int((singular > singular[0] * 1.0e-4).sum()),
        "finite_difference": fd_rows,
        "median_relative_error": float(np.median([row["relative_error"] for row in fd_rows])),
        "max_relative_error": float(max(row["relative_error"] for row in fd_rows)),
    }


def _collect_executed_context(
    env: gym.Env,
    *,
    seed: int,
    history_len: int,
    action_block: int,
    warmup_packets: np.ndarray | None = None,
) -> dict[str, Any]:
    """Create an RGB/action prefix by executing full, non-padding packets."""

    if history_len < 1 or action_block < 1:
        raise ValueError("TwoRoom history length and action block must be positive")
    _observation, info = env.reset(seed=int(seed))
    goal = np.asarray(info["goal"], dtype=np.uint8).copy()
    low, high = np.asarray(env.action_space.low, dtype=np.float32), np.asarray(env.action_space.high, dtype=np.float32)
    if low.shape != high.shape or not np.all(low < high):
        raise ValueError("TwoRoom has an invalid environment action box")
    if warmup_packets is None:
        raise ValueError("TwoRoom D2P requires source-supported executed warmup packets")
    packets = np.asarray(warmup_packets, dtype=np.float32)
    if packets.shape != (history_len, action_block, low.size) or not np.isfinite(packets).all():
        raise ValueError("TwoRoom support warmup packets have an incompatible full-packet layout")
    if np.any(packets < low[None, None] - 1e-6) or np.any(packets > high[None, None] + 1e-6):
        raise ValueError("TwoRoom support warmup packets leave the live action box")
    frames: list[np.ndarray] = []
    for packet in packets:
        frames.append(np.asarray(env.render(), dtype=np.uint8).copy())
        for action in packet:
            _observation, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                raise RuntimeError("TwoRoom warmup ended the episode before the planning root")
    root = np.asarray(env.render(), dtype=np.uint8).copy()
    return {
        "frames": frames,
        "history_raw_packets": packets,
        "root_frame": root,
        "goal": goal,
        "context_mode": "executed_full_packet_warmup",
        "warmup_policy": WARMUP_POLICY,
        "warmup_steps": int(history_len * action_block),
        "history_rgb_sha256": [_sha256_array(frame) for frame in frames],
        "root_rgb_sha256": _sha256_array(root),
        "goal_rgb_sha256": _sha256_array(goal),
        "warmup_actions_sha256": _sha256_array(packets),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy", type=Path, default=Path("/home/gbk/.stable_worldmodel/checkpoints/tworoom/lewm"))
    parser.add_argument(
        "--training-action-stats", type=Path,
        default=Path("results/tworoom_d2p_training_action_stats_20261024/tworoom_training_action_stats.json"),
    )
    parser.add_argument("--training-dataset", type=Path, default=Path("/home/gbk/.stable_worldmodel/datasets/tworoom.h5"))
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--roots", type=int, default=4)
    parser.add_argument("--seed", type=int, default=202610250)
    parser.add_argument("--history-len", type=int, default=3)
    parser.add_argument("--future-horizon", type=int, default=2)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--probe-dims", type=int, default=8)
    parser.add_argument("--fd-pairs", type=int, default=8)
    parser.add_argument("--fd-epsilon", type=float, default=1.0e-3)
    args = parser.parse_args()
    if min(args.roots, args.history_len, args.future_horizon, args.action_block, args.probe_dims, args.fd_pairs) < 1:
        parser.error("root counts, horizons, action block, and probe counts must be positive")
    if args.fd_epsilon <= 0.0:
        parser.error("fd-epsilon must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()
    args.policy = args.policy.resolve()
    args.training_action_stats = args.training_action_stats.resolve()
    args.training_dataset = args.training_dataset.resolve()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable TwoRoom D2P P0 output: {args.output}")
    runtime = require_cuda_allocation(args.device)
    device = torch.device(args.device)
    stat_row = validate_training_semantic_receipt(args.training_action_stats)
    mean = np.asarray(stat_row["calibration_action_mean"], dtype=np.float32)
    std = np.asarray(stat_row["calibration_action_std"], dtype=np.float32)
    receipt_dataset, receipt_sha256 = packet_support.receipt_source(args.training_action_stats)
    if args.training_dataset != receipt_dataset.resolve():
        raise ValueError("TwoRoom P0 training dataset differs from the source locked by the statistic receipt")
    model = load_compatible_pretrained(args.policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    parameter_before = _parameter_hash(model)
    env = TwoRoomGoalAliasWrapper(gym.make("swm/TwoRoom-v1", render_mode="rgb_array"))
    action_dim = int(np.prod(env.action_space.shape))
    checkpoint_block = _validate_tworoom_action_protocol(
        model, mean, std, action_dim=action_dim, action_block=int(args.action_block),
    )
    lower = torch.as_tensor(np.asarray(env.action_space.low, dtype=np.float32), device=device)
    upper = torch.as_tensor(np.asarray(env.action_space.high, dtype=np.float32), device=device)
    transform = _image_transform()
    support_pool, support_meta = packet_support.load_packet_pool(
        args.training_dataset, expected_sha256=receipt_sha256, action_block=int(args.action_block), action_dim=action_dim,
    )
    roots: list[dict[str, Any]] = []
    started = time.time()
    try:
        for root_index in range(int(args.roots)):
            seed = int(args.seed + root_index)
            warmup, warmup_indices = packet_support.warmup_packets(support_pool, history_len=int(args.history_len), seed=seed)
            context = _collect_executed_context(env, seed=seed, history_len=int(args.history_len), action_block=int(args.action_block), warmup_packets=warmup)
            audit = _jacobian_and_fd(
                model, context["frames"], context["root_frame"], context["history_raw_packets"], context["goal"],
                mean=mean, std=std, action_block=int(args.action_block), action_dim=action_dim,
                future_horizon=int(args.future_horizon), probe_dims=int(args.probe_dims),
                fd_pairs=int(args.fd_pairs), epsilon=float(args.fd_epsilon), seed=seed,
                transform=transform, device=device, low=lower, high=upper,
            )
            roots.append({
                "seed": seed,
                "jacobian": audit,
                "executed_context": {key: context[key] for key in (
                    "context_mode", "warmup_policy", "warmup_steps", "history_rgb_sha256",
                    "root_rgb_sha256", "goal_rgb_sha256", "warmup_actions_sha256",
                )} | {"training_packet_indices": warmup_indices.tolist()},
            })
            print(json.dumps({"seed": seed, "jacobian_norm": audit["frobenius_norm"], "fd_median_relative_error": audit["median_relative_error"]}, sort_keys=True))
    finally:
        env.close()
    parameter_after = _parameter_hash(model)
    norms = [float(row["jacobian"]["frobenius_norm"]) for row in roots]
    fd_errors = [float(row["jacobian"]["median_relative_error"]) for row in roots]
    gates = {
        "cuda_only": True,
        "cuda_real_allocation": True,
        "frozen_parameter_hash_unchanged": parameter_before == parameter_after,
        "checkpoint_action_block_matches_requested": int(checkpoint_block) == int(args.action_block),
        "training_semantic_tiled_action_statistics": True,
        "executed_warmup_packets_are_source_supported": True,
        "executed_full_packet_history_no_padding": True,
        "every_root_nonzero_finite_future_action_jacobian": bool(norms and min(norms) > 1.0e-10),
        "central_fd_median_relative_error_le_0_10_each_root": bool(fd_errors and max(fd_errors) <= 0.10),
        "planner_input_is_rendered_rgb_goal_rgb_and_causal_actions_only": True,
        "evaluator_state_not_used_as_model_input": True,
        "no_adapter_teacher_or_privileged_ranker": True,
        "p5_outcome_roots_not_opened": True,
    }
    compat = getattr(model, "_checkpoint_compat", {})
    weights = Path(str(compat.get("weights", "")))
    report = {
        "schema": SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "scope": "CUDA-only frozen TwoRoom D2P P0 input/Jacobian admission; no physical planning, reconfiguration, composition, or P5 efficacy conclusion.",
        "cuda_host_preflight": {**runtime, "python": platform.python_version()},
        "inputs": {
            "checkpoint_weights": str(weights),
            "checkpoint_weights_sha256": _sha256_file(weights) if weights.is_file() else None,
            "checkpoint_remapped_keys": int(compat.get("remapped_keys", -1)),
            "training_action_statistics": str(args.training_action_stats),
            "training_action_statistics_sha256": _sha256_file(args.training_action_stats),
            "training_packet_support": support_meta,
            "raw_environment_action_box": [lower.detach().cpu().tolist(), upper.detach().cpu().tolist()],
        },
        "protocol": {
            "environment": "swm/TwoRoom-v1",
            "runtime_input": "rendered RGB/action causal history ending at an observed RGB root, rendered goal RGB, and bounded future raw actions",
            "observed_root_role": "evaluator-side RGB identity anchor for the causal prefix; no state vector is supplied to the model",
            "action_packet": "five source-supported, independently executed raw [x,y] controls, flattened time-major after raw-action z-score",
            "history_len": int(args.history_len),
            "future_horizon": int(args.future_horizon),
            "action_block": int(args.action_block),
            "checkpoint_action_block": int(checkpoint_block),
            "root_seeds": [row["seed"] for row in roots],
            "future_stages": "A root-disjoint P1 calibration/outcome protocol may be written only after this P0 receipt passes; this run reserves no outcome labels.",
        },
        "parameter_hash_before": parameter_before,
        "parameter_hash_after": parameter_after,
        "gates": gates,
        "aggregate": {"jacobian_frobenius_norms": norms, "fd_median_relative_errors": fd_errors},
        "roots": roots,
        "elapsed_seconds": time.time() - started,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "gates": gates, "output": str(args.output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
