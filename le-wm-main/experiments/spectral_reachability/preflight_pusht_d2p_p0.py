"""CUDA-only, RGB-only D2P-P0 admission for the published frozen PushT LeWM.

This is deliberately *not* a PushT efficacy runner.  It establishes the
implementation contract needed before asking whether a Planning Compiler can
help on contact-rich PushT: the frozen model must expose a non-zero,
finite, action-to-predicted-latent derivative whose selected entries agree
with central finite differences.  The planner sees only a rendered current
RGB frame, a rendered goal RGB frame, and bounded candidate actions.  The
simulator state is used only by the evaluator to recreate deterministic roots
and render those two images; it is never supplied to the model, score, or
action compiler.

In particular this runner must not be confused with the older PushT oracle,
adapter, decoder, or state-heuristic experiments.  It loads no learned
component besides the frozen published LeWM and it creates no outcome roots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401 -- registers swm/PushT-v1
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_pusht_reachability_horizon import collect_nonterminal_boundary_contact_roots
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.gpu_preflight import require_cuda_allocation
from experiments.online_planning_protocol import validate_action_protocol
from experiments.pusht_protocol_contracts import root_fingerprint
from experiments.run_pusht_counterfactual import (
    PUBLIC_ACTION_MEAN,
    PUBLIC_ACTION_STD,
    image_transform,
)


SCHEMA = "pusht_d2p_p0_preflight.v1"


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


def _bounded_live_actions(params: torch.Tensor) -> torch.Tensor:
    """Map unconstrained compiler coordinates to PushT's legal live box."""

    if params.ndim != 2 or params.shape[1] != 2:
        raise ValueError("PushT compiler parameters must have shape [horizon, 2]")
    return torch.tanh(params)


def _model_action_tokens(live_actions: torch.Tensor, *, action_block: int) -> torch.Tensor:
    """Differentiate through the released PushT action-normalization contract.

    A live PushT action is a relative displacement in ``[-1, 1]``.  Released
    LeWM data first divides it by 100 and then applies the public two-channel
    z-score; each model token contains ``action_block`` repeated time-major
    actions.  Keeping this transform in torch is essential: a numpy conversion
    here would make a zero Jacobian look like a model limitation.
    """

    if live_actions.ndim != 2 or live_actions.shape[1] != 2 or action_block < 1:
        raise ValueError("invalid live action or action-block layout")
    mean = torch.as_tensor(PUBLIC_ACTION_MEAN, dtype=live_actions.dtype, device=live_actions.device)
    std = torch.as_tensor(PUBLIC_ACTION_STD, dtype=live_actions.dtype, device=live_actions.device)
    normalized = (live_actions / 100.0 - mean) / std
    return normalized.repeat_interleave(int(action_block), dim=1)


def _collect_causal_rgb_context(
    env: gym.Env, root: dict[str, np.ndarray], *, history_len: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Collect a causal zero-action RGB/action history and a goal image.

    The goal image is rendered by resetting a private evaluator copy to the
    declared goal pose.  Only the resulting RGB array crosses into the model
    path; no state vector, geometry feature, or oracle successor is retained.
    """

    if history_len < 1:
        raise ValueError("history_len must be positive")
    state = np.asarray(root["state"], dtype=np.float64).copy()
    goal = np.asarray(root["goal"], dtype=np.float64).copy()
    env.reset(seed=0, options={"state": state, "goal_state": goal})
    # A zero live action is a causal, legal action token.  It is not padding:
    # each of the H source frames is rendered before a real simulator step.
    # Its only job here is to expose the frozen rollout API's required
    # observed action prefix before any future action is differentiated.
    history_actions = np.zeros((int(history_len), 2), dtype=np.float32)
    frames: list[np.ndarray] = []
    for action in history_actions:
        frames.append(np.asarray(env.render(), dtype=np.uint8).copy())
        _observation, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            raise RuntimeError("causal PushT warmup reached a terminal state")
    env.reset(seed=0, options={"state": goal, "goal_state": goal})
    goal_rgb = np.asarray(env.render(), dtype=np.uint8).copy()
    if len(frames) != history_len or any(frame.ndim != 3 for frame in frames) or goal_rgb.shape != frames[0].shape or frames[0].shape[-1] != 3:
        raise RuntimeError("PushT evaluator did not produce paired RGB source/goal frames")
    return frames, history_actions, goal_rgb


def _image_tensor(image: np.ndarray, *, transform: Any, device: torch.device) -> torch.Tensor:
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("expected uint8 HWC RGB image")
    # ``LeWM.encode`` consumes [batch, time, channel, height, width].  The
    # rollout path adds its own decision/horizon axis below.  Adding that axis
    # here would turn a one-frame goal into six dimensions and make the ViT
    # error before any action derivative is evaluated.
    return transform(torch.from_numpy(image).permute(2, 0, 1)).to(device)[None, None]


def _frames_tensor(frames: list[np.ndarray], *, transform: Any, device: torch.device) -> torch.Tensor:
    if not frames:
        raise ValueError("at least one causal RGB frame is required")
    processed = [_image_tensor(frame, transform=transform, device=device)[0, 0] for frame in frames]
    return torch.stack(processed)[None]


def _terminal_prediction(
    model: torch.nn.Module,
    context_frames: list[np.ndarray],
    history_actions: np.ndarray,
    params: torch.Tensor,
    *,
    transform: Any,
    device: torch.device,
    action_block: int,
) -> torch.Tensor:
    """Predict the terminal latent under a differentiable bounded action chunk."""

    pixels = _frames_tensor(context_frames, transform=transform, device=device)
    history = torch.as_tensor(history_actions, dtype=torch.float32, device=device)
    if pixels.shape[1] != history.shape[0]:
        raise ValueError("causal RGB/action history lengths differ")
    future_tokens = _model_action_tokens(_bounded_live_actions(params), action_block=action_block)
    history_tokens = _model_action_tokens(history, action_block=action_block)
    action_sequence = torch.cat((history_tokens, future_tokens), dim=0)[None, None]
    rollout_pixels = pixels[:, None]
    predicted = model.rollout({"pixels": rollout_pixels}, action_sequence, history_size=int(pixels.shape[1]))
    terminal = predicted["predicted_emb"][0, 0, -1]
    if terminal.ndim != 1 or not torch.isfinite(terminal).all():
        raise RuntimeError("frozen PushT model produced an invalid terminal latent")
    return terminal


def _goal_latent(
    model: torch.nn.Module, goal_rgb: np.ndarray, *, transform: Any, device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        goal = model.encode({"pixels": _image_tensor(goal_rgb, transform=transform, device=device)})["emb"][0, 0]
    if goal.ndim != 1 or not torch.isfinite(goal).all():
        raise RuntimeError("frozen PushT model produced an invalid goal latent")
    return goal.detach()


def _jacobian_and_fd(
    model: torch.nn.Module,
    context_frames: list[np.ndarray],
    history_actions: np.ndarray,
    goal_rgb: np.ndarray,
    *,
    transform: Any,
    device: torch.device,
    horizon: int,
    action_block: int,
    probe_dims: int,
    fd_pairs: int,
    epsilon: float,
    seed: int,
) -> dict[str, Any]:
    """Audit the goal-relative action Jacobian with deterministic FD probes."""

    if horizon < 1 or probe_dims < 1 or fd_pairs < 1 or epsilon <= 0:
        raise ValueError("invalid P0 Jacobian/finite-difference settings")
    goal = _goal_latent(model, goal_rgb, transform=transform, device=device)
    origin = torch.zeros((horizon, 2), dtype=torch.float32, device=device, requires_grad=True)
    with torch.no_grad():
        latent_dim = int(_terminal_prediction(
            model, context_frames, history_actions, origin.detach(), transform=transform, device=device, action_block=action_block,
        ).numel())
    generator = torch.Generator(device=device).manual_seed(int(seed))
    projection = torch.randn((min(int(probe_dims), latent_dim), latent_dim), generator=generator, device=device)
    projection = projection / projection.norm(dim=1, keepdim=True).clamp_min(1.0e-8)

    def projected(flat: torch.Tensor) -> torch.Tensor:
        terminal = _terminal_prediction(
            model, context_frames, history_actions, flat.reshape(horizon, 2), transform=transform, device=device, action_block=action_block,
        )
        # The scalar/vector objective is goal-relative, while the derivative
        # remains entirely model-native; the goal is supplied as RGB only.
        return projection @ (goal - terminal)

    jacobian = torch.autograd.functional.jacobian(projected, origin.reshape(-1), vectorize=True)
    if jacobian.ndim != 2 or not torch.isfinite(jacobian).all() or float(jacobian.norm()) <= 1.0e-10:
        raise RuntimeError("future-action Jacobian is non-finite or zero")
    fd_rows: list[dict[str, float | int]] = []
    for pair in range(min(int(fd_pairs), int(jacobian.shape[0]), int(jacobian.shape[1]))):
        output_index = pair % int(jacobian.shape[0])
        input_index = (3 * pair + 1) % int(jacobian.shape[1])
        plus, minus = origin.detach().reshape(-1).clone(), origin.detach().reshape(-1).clone()
        plus[input_index] += float(epsilon)
        minus[input_index] -= float(epsilon)
        with torch.no_grad():
            central = float((projected(plus)[output_index] - projected(minus)[output_index]) / (2.0 * epsilon))
        autograd = float(jacobian[output_index, input_index])
        absolute = abs(autograd - central)
        fd_rows.append({
            "output_index": output_index,
            "action_parameter_index": input_index,
            "autograd": autograd,
            "central_fd": central,
            "absolute_error": absolute,
            "relative_error": absolute / max(abs(autograd), abs(central), 1.0e-6),
        })
    singular = torch.linalg.svdvals(jacobian)
    return {
        "action_layout": {
            "live_action_shape": [int(horizon), 2],
            "model_action_block": int(action_block),
            "model_token_width": int(2 * action_block),
            "causal_history_rows": int(len(context_frames)),
            "model_horizon": int(horizon),
            "parameter_count": int(jacobian.shape[1]),
            "time_major_block_repeat": True,
        },
        "jacobian_shape": [int(item) for item in jacobian.shape],
        "frobenius_norm": float(jacobian.norm()),
        "singular_values": [float(value) for value in singular],
        "effective_rank_1e-4": int((singular > singular[0] * 1.0e-4).sum()),
        "finite_difference": fd_rows,
        "median_relative_error": float(np.median([row["relative_error"] for row in fd_rows])),
        "max_relative_error": float(max(row["relative_error"] for row in fd_rows)),
    }


def _reserved_outcome_contract(*, root_seed: int, root_count: int) -> dict[str, Any]:
    """Declare outcome provenance without generating, rendering, or scoring it."""

    return {
        "status": "reserved_not_opened",
        "root_seed": int(root_seed),
        "root_count": int(root_count),
        "oversample_factor": 4,
        "model_calls": 0,
        "backward_calls": 0,
        "rule": "Outcome roots must not be generated, rendered, scored, or used for calibration before a separately sealed P1 admission receipt passes.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy", default="pusht/lewm")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--calibration-roots", type=int, default=4)
    parser.add_argument("--calibration-seed", type=int, default=202610241)
    parser.add_argument("--reserved-outcome-roots", type=int, default=24)
    parser.add_argument("--reserved-outcome-seed", type=int, default=202610281)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--history-len", type=int, default=3)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--probe-dims", type=int, default=8)
    parser.add_argument("--fd-pairs", type=int, default=8)
    parser.add_argument("--fd-epsilon", type=float, default=1.0e-3)
    args = parser.parse_args()
    if min(args.calibration_roots, args.reserved_outcome_roots, args.horizon, args.history_len, args.action_block, args.probe_dims, args.fd_pairs) < 1:
        parser.error("root counts, horizon, action block, and probe counts must be positive")
    if args.fd_epsilon <= 0:
        parser.error("fd-epsilon must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable D2P P0 output: {args.output}")
    if args.device != "cuda":  # argparse is intentionally redundant with this receipt-level invariant.
        raise RuntimeError("PushT D2P-P0 is CUDA-only; CPU fallback is forbidden")
    runtime = require_cuda_allocation(args.device)
    device = torch.device(args.device)
    model = load_compatible_pretrained(str(args.policy), device=device).eval()
    model.requires_grad_(False)
    checkpoint_block = validate_action_protocol(model, action_dim=2, action_block=int(args.action_block))
    parameter_before = _parameter_hash(model)
    transform = image_transform()
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    started = time.time()
    try:
        roots = collect_nonterminal_boundary_contact_roots(
            env, count=int(args.calibration_roots), seed=int(args.calibration_seed), oversample_factor=4,
        )
        rows: list[dict[str, Any]] = []
        for root_id, root in enumerate(roots):
            context_frames, history_actions, goal_rgb = _collect_causal_rgb_context(
                env, root, history_len=int(args.history_len),
            )
            audit = _jacobian_and_fd(
                model, context_frames, history_actions, goal_rgb, transform=transform, device=device,
                horizon=int(args.horizon), action_block=int(args.action_block),
                probe_dims=int(args.probe_dims), fd_pairs=int(args.fd_pairs),
                epsilon=float(args.fd_epsilon), seed=int(args.calibration_seed + root_id),
            )
            rows.append({
                "root_id": int(root_id),
                "root_sha256": root_fingerprint(root),
                "causal_history_rgb_sha256": hashlib.sha256(np.stack(context_frames).tobytes()).hexdigest(),
                "causal_history_action_sha256": hashlib.sha256(history_actions.tobytes()).hexdigest(),
                "goal_rgb_sha256": hashlib.sha256(goal_rgb.tobytes()).hexdigest(),
                "jacobian": audit,
            })
            print(json.dumps({"root_id": root_id, "jacobian_norm": audit["frobenius_norm"], "fd_median_relative_error": audit["median_relative_error"]}, sort_keys=True))
    finally:
        env.close()
    parameter_after = _parameter_hash(model)
    norms = [float(row["jacobian"]["frobenius_norm"]) for row in rows]
    fd_errors = [float(row["jacobian"]["median_relative_error"]) for row in rows]
    gates = {
        "cuda_only": True,
        "cuda_real_allocation": True,
        "frozen_parameter_hash_unchanged": parameter_before == parameter_after,
        "checkpoint_action_block_matches_requested": int(checkpoint_block) == int(args.action_block),
        "every_root_nonzero_finite_action_jacobian": bool(norms and min(norms) > 1.0e-10),
        "central_fd_median_relative_error_le_0_10_each_root": bool(fd_errors and max(fd_errors) <= 0.10),
        "planner_input_is_rgb_goal_rgb_and_action_only": True,
        "no_oracle_successor_or_state_in_planning_score": True,
        "no_adapter_repaired_predictor_or_privileged_ranker": True,
        "outcome_roots_not_opened": True,
    }
    compat = getattr(model, "_checkpoint_compat", {})
    weights = Path(str(compat.get("weights", "")))
    report = {
        "schema": SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "scope": "CUDA-only frozen PushT D2P P0 implementation admission; it does not establish model fidelity, physical planning efficacy, reconfiguration, or P5 transfer.",
        "cuda_host_preflight": {**runtime, "python": platform.python_version()},
        "inputs": {
            "checkpoint_weights": str(weights),
            "checkpoint_weights_sha256": _sha256_file(weights) if weights.is_file() else None,
            "checkpoint_remapped_keys": int(compat.get("remapped_keys", -1)),
            "action_normalization": {
                "live_action_box": [-1.0, 1.0],
                "raw_action_divisor": 100.0,
                "mean": PUBLIC_ACTION_MEAN.tolist(),
                "std": PUBLIC_ACTION_STD.tolist(),
                "model_token_layout": "time-major repeated [normalized_ax, normalized_ay]",
            },
        },
        "protocol": {
            "environment": "swm/PushT-v1",
            "runtime_input": "rendered current RGB, rendered goal RGB, and bounded live action parameters only",
            "evaluator_only_state_use": "deterministic root reconstruction and RGB rendering only",
            "simulator_state_used_as_model_input": False,
            "oracle_successor_used_in_planner": False,
            "learned_adapter_or_repaired_predictor": False,
            "privileged_state_ranker": False,
            "calibration_roots": int(args.calibration_roots),
            "calibration_root_seed": int(args.calibration_seed),
            "root_protocol": "unique_nonterminal_boundary_contact",
            "horizon": int(args.horizon),
            "history_len": int(args.history_len),
            "action_block": int(args.action_block),
            "checkpoint_action_block": int(checkpoint_block),
            "outcome": _reserved_outcome_contract(root_seed=int(args.reserved_outcome_seed), root_count=int(args.reserved_outcome_roots)),
            "future_comparison_budget_contract": {
                "required_arms": ["CEM", "MPPI", "vanilla_action_GD", "local_Gauss_Newton", "GRASP_style", "SRA_without_composition", "full_D2P"],
                "must_record": ["model_forward_calls", "backward_calls", "candidate_count", "wall_time_seconds"],
            },
        },
        "parameter_hash_before": parameter_before,
        "parameter_hash_after": parameter_after,
        "gates": gates,
        "aggregate": {"jacobian_frobenius_norms": norms, "fd_median_relative_errors": fd_errors},
        "calibration_roots": rows,
        "elapsed_seconds": time.time() - started,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "gates": gates, "output": str(args.output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
