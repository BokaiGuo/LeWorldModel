"""Independent RGB-only closed-loop consequence for Cube temporal ranking.

The readouts are fit on calibration roots only.  At deployment each arm gets
only the live current/previous RGB frame, rendered goal, previous action, and
the fixed candidate action bank.  A fixed, non-adaptive descent/close prelude
is shared by both arms; after the prelude each arm replans one action at a
time.  Privileged state is used only to restore identical initial conditions
between paired arms and to score final goal distance.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Import PyTorch before configuring MuJoCo's headless renderer.  In the
# managed CUDA runtime, exporting MUJOCO_GL before importing PyTorch can make
# the CUDA/NVML availability probe report zero devices.  Configuring EGL
# inside the process after PyTorch is loaded keeps CUDA visible while making
# RGB rendering deterministic and headless.
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import stable_worldmodel  # noqa: F401 - registers environments

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_cube_rgb_observability import (  # noqa: E402
    extract_color_geometry_features,
    extract_color_geometry_temporal_features,
    _target_red_centroid,
    _goal_position,
    _make_env,
    _vec,
)
from experiments.audit_cube_temporal_action_effect import (  # noqa: E402
    build_feature_families,
    collect_temporal_frames,
)
from experiments.audit_cube_temporal_action_ranking import (  # noqa: E402
    _candidate_bank,
    _fit_readout,
    _predict,
)
from experiments.checkpoint_compat import load_compatible_pretrained  # noqa: E402


def _capture_initial(seed: int, task_id: int) -> dict[str, Any]:
    env = _make_env()
    try:
        _, info = env.reset(seed=int(seed), options={"task_id": int(task_id), "render_goal": True})
        raw = env.unwrapped
        return {
            "seed": int(seed),
            "task_id": int(task_id),
            "variation_values": copy.deepcopy(raw.variation_space.value),
            "state": np.concatenate([raw.data.qpos.copy(), raw.data.qvel.copy()]).tolist(),
            "goal": _goal_position(env)[:2].copy(),
        }
    finally:
        env.close()


def _prelude_action(step: int) -> np.ndarray:
    action = np.zeros(5, dtype=np.float32)
    if step < 15:
        action[2] = -1.0
    elif step < 20:
        # OGB-Cube uses relative gripper-opening actions: +1 closes.
        action[4] = 1.0
    return action


def _support_exploration_action(step: int, candidates: np.ndarray) -> np.ndarray:
    """Fixed RGB-only fallback when the candidate utility is not identifiable."""
    indices = (1, 2, 3, 4)  # x+, x-, y+, y-
    return np.asarray(candidates[indices[step % len(indices)]], dtype=np.float32)


def _rgb_block_goal_error(current: np.ndarray, goal: np.ndarray) -> tuple[float, bool]:
    """Return RGB block-to-goal error norm for a deployment-only feedback gate."""
    current_xy = _target_red_centroid(current)[:2]
    goal_xy = _target_red_centroid(goal)[:2]
    valid = bool(current_xy.shape == (2,) and goal_xy.shape == (2,))
    valid = valid and bool(np.linalg.norm(current_xy) > 0.0 or np.linalg.norm(goal_xy) > 0.0)
    if not valid:
        return 0.0, False
    return float(np.linalg.norm(goal_xy - current_xy)), True


def _robust_rank_score(scores: torch.Tensor) -> torch.Tensor:
    """Normalize candidate costs without trusting either scale or variance.

    Latent and RGB readouts live on different regression scales.  A direct
    sum would therefore let one view dominate for purely numerical reasons.
    Median/IQR normalization is fixed at deployment and uses only the current
    candidate bank; it cannot inspect evaluator state or test outcomes.
    """
    if scores.ndim != 1 or scores.numel() < 2:
        raise ValueError("rank normalization expects a one-dimensional candidate vector")
    median = scores.median()
    quantiles = torch.tensor([0.25, 0.75], device=scores.device, dtype=scores.dtype)
    q25, q75 = torch.quantile(scores, quantiles)
    scale = (q75 - q25).clamp_min(1.0e-6)
    return (scores - median) / scale


def _rank_consensus_score(
    latent_scores: torch.Tensor, rgb_scores: torch.Tensor,
) -> torch.Tensor:
    """Return a fixed two-view rank-consensus cost for one candidate bank."""
    if latent_scores.shape != rgb_scores.shape:
        raise ValueError("latent and RGB candidate score shapes must match")
    return _robust_rank_score(latent_scores) + _robust_rank_score(rgb_scores)


def _run_arm(
    *, initial_conditions: list[dict[str, Any]], model: torch.nn.Module,
    fit: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    candidates: np.ndarray,
    device: str, max_steps: int, prelude_steps: int, batch_size: int,
    method: str,
    support_gate: bool = False,
    support_threshold: float = 5.0e-4,
    pixel_feedback: bool = False,
    pixel_feedback_threshold: float = 1.0e-3,
    pixel_feedback_mode: str = "persistent",
) -> dict[str, Any]:
    if pixel_feedback_mode not in {"persistent", "one_step"}:
        raise ValueError(f"unsupported pixel feedback mode: {pixel_feedback_mode}")
    episodes = []
    for initial in initial_conditions:
        env = _make_env()
        try:
            obs, info = env.reset(
                seed=int(initial["seed"]),
                options={
                    "task_id": int(initial["task_id"]),
                    "render_goal": True,
                    "variation_values": copy.deepcopy(initial["variation_values"]),
                    "state": np.asarray(initial["state"], dtype=np.float64),
                },
            )
            goal_image = np.asarray(env.unwrapped._cur_goal_rendered, dtype=np.uint8).copy()
            current = np.asarray(obs, dtype=np.uint8).copy()
            previous = current.copy()
            previous_action = np.zeros(5, dtype=np.float32)
            actions: list[list[float]] = []
            distances: list[float] = []
            fallback_count = 0
            pixel_feedback_sign = 1.0
            pixel_feedback_reversals = 0
            last_pixel_error_norm: float | None = None
            pixel_error_trace: list[float] = []
            pixel_feedback_sign_trace: list[float] = []
            for step in range(max_steps):
                reverse_current_pulse = False
                current_pixel_error, current_pixel_valid = _rgb_block_goal_error(current, goal_image)
                if current_pixel_valid:
                    pixel_error_trace.append(current_pixel_error)
                    if step == prelude_steps:
                        pixel_feedback_sign = 1.0
                        last_pixel_error_norm = current_pixel_error
                    elif pixel_feedback and step > prelude_steps and last_pixel_error_norm is not None:
                        if current_pixel_error > last_pixel_error_norm + pixel_feedback_threshold:
                            if pixel_feedback_mode == "persistent":
                                pixel_feedback_sign *= -1.0
                            else:
                                reverse_current_pulse = True
                            pixel_feedback_reversals += 1
                        last_pixel_error_norm = current_pixel_error
                if step < prelude_steps:
                    action = _prelude_action(step)
                else:
                    n_candidates = len(candidates)
                    current_batch = np.repeat(current[None], n_candidates, axis=0)
                    previous_batch = np.repeat(previous[None], n_candidates, axis=0)
                    goal_batch = np.repeat(goal_image[None], n_candidates, axis=0)
                    previous_actions = np.repeat(previous_action[None], n_candidates, axis=0)
                    valid = np.ones(n_candidates, dtype=np.float32)
                    families, _ = build_feature_families(
                        current_batch, previous_batch, goal_batch, candidates,
                        previous_actions, valid, device=device, model=model,
                        batch_size=batch_size,
                    )
                    if method in {"rgb_geometry_action", "latent_rgb_rank_consensus"}:
                        families["rgb_geometry_action"] = extract_color_geometry_features(
                            current_batch, goal_batch, candidates, device,
                        )["color_geometry_action"]
                    elif method == "rgb_geometry_temporal_action":
                        families[method] = extract_color_geometry_temporal_features(
                            current_batch, previous_batch, goal_batch, candidates,
                            previous_actions, device,
                        )["color_geometry_temporal_action"]
                    if method == "latent_rgb_rank_consensus":
                        if not isinstance(fit, dict):
                            raise TypeError("rank consensus requires both latent and RGB fits")
                        latent = _predict(families["latent_temporal"], fit["latent_temporal"])
                        rgb = _predict(families["rgb_geometry_action"], fit["rgb_geometry_action"])
                        latent_scores = torch.linalg.vector_norm(latent, dim=1)
                        rgb_scores = torch.linalg.vector_norm(rgb, dim=1)
                        predicted_scores = _rank_consensus_score(latent_scores, rgb_scores)
                    else:
                        if not isinstance(fit, tuple):
                            raise TypeError(f"single-view method {method} requires one fit")
                        predicted = _predict(families[method], fit)
                        predicted_scores = torch.linalg.vector_norm(predicted, dim=1)
                    spread = float((predicted_scores.max() - predicted_scores.min()).item())
                    if support_gate and spread < support_threshold:
                        action = _support_exploration_action(step - prelude_steps, candidates)
                        fallback_count += 1
                    else:
                        action = candidates[int(predicted_scores.argmin().item())]
                    should_reverse = (
                        pixel_feedback
                        and (
                            (pixel_feedback_mode == "persistent" and pixel_feedback_sign < 0.0)
                            or (pixel_feedback_mode == "one_step" and reverse_current_pulse)
                        )
                    )
                    if should_reverse:
                        action = np.asarray(action, dtype=np.float32).copy()
                        action[:2] *= -1.0
                    pixel_feedback_sign_trace.append(
                        -1.0 if should_reverse else 1.0
                    )
                if pixel_feedback and step < prelude_steps:
                    pixel_feedback_sign_trace.append(float(pixel_feedback_sign))
                actions.append(np.asarray(action, dtype=np.float32).tolist())
                next_obs, _, terminated, truncated, next_info = env.step(action)
                block = _vec(next_info, "privileged/block_0_pos", 3)[:2]
                distances.append(float(np.linalg.norm(block - initial["goal"])))
                previous, current = current, np.asarray(next_obs, dtype=np.uint8).copy()
                previous_action = np.asarray(action, dtype=np.float32).copy()
                info = next_info
                if terminated or truncated:
                    break
            final_block = _vec(info, "privileged/block_0_pos", 3)[:2]
            final_distance = float(np.linalg.norm(final_block - initial["goal"]))
            episodes.append({
                "seed": int(initial["seed"]),
                "task_id": int(initial["task_id"]),
                "steps": len(actions),
                "final_goal_distance_m": final_distance,
                "best_goal_distance_m": float(np.min(distances)) if distances else None,
                "actions": actions,
                "support_fallback_count": int(fallback_count),
                "pixel_feedback_enabled": bool(pixel_feedback),
                "pixel_feedback_mode": pixel_feedback_mode,
                "pixel_feedback_reversal_count": int(pixel_feedback_reversals),
                "pixel_error_trace_runtime": pixel_error_trace,
                "pixel_feedback_sign_trace_runtime": pixel_feedback_sign_trace,
            })
        finally:
            env.close()
    return {
        "method": method,
        "episodes": episodes,
        "mean_final_goal_distance_m": float(np.mean([row["final_goal_distance_m"] for row in episodes])),
        "mean_best_goal_distance_m": float(np.mean([row["best_goal_distance_m"] for row in episodes])),
        "support_fallback_rate": float(np.mean([row["support_fallback_count"] for row in episodes]) / max(max_steps - prelude_steps, 1)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", default="/home/gbk/.stable_worldmodel/checkpoints/cube/lewm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261070)
    parser.add_argument("--calibration-roots", type=int, default=40)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--prelude-steps", type=int, default=20)
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--support-threshold", type=float, default=5.0e-4)
    parser.add_argument("--pixel-feedback-threshold", type=float, default=1.0e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for Cube temporal closed loop but unavailable")
    if args.episodes < 4 or args.calibration_roots < 8:
        raise ValueError("need at least 4 episodes and 8 calibration roots")
    if args.prelude_steps < 2 or args.max_steps <= args.prelude_steps:
        raise ValueError("max-steps must exceed prelude-steps >= 2")
    if args.support_threshold < 0.0 or not np.isfinite(args.support_threshold):
        raise ValueError("support-threshold must be finite and non-negative")
    if args.pixel_feedback_threshold < 0.0 or not np.isfinite(args.pixel_feedback_threshold):
        raise ValueError("pixel-feedback-threshold must be finite and non-negative")
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        "[purpose] independent RGB-only Cube closed-loop consequence; "
        f"device={args.device} episodes={args.episodes}"
    )
    model = load_compatible_pretrained(args.policy, device=args.device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    cal_current, cal_previous, cal_goals, cal_previous_actions, cal_labels = collect_temporal_frames(
        roots=args.calibration_roots, frames_per_root=24, tasks=args.tasks,
        seed=args.seed, action_seed=args.seed + 1, collection_mode="guided_intervention",
    )
    cal_families, _ = build_feature_families(
        cal_current, cal_previous, cal_goals, cal_labels["action"], cal_previous_actions,
        cal_labels["history_valid"], device=args.device, model=model, batch_size=args.batch_size,
    )
    cal_target = torch.as_tensor(
        cal_labels["next_block_goal_delta_xy"], device=args.device, dtype=torch.float32,
    )
    cal_color = extract_color_geometry_features(
        cal_current, cal_goals, cal_labels["action"], args.device,
    )
    cal_color_temporal = extract_color_geometry_temporal_features(
        cal_current, cal_previous, cal_goals, cal_labels["action"],
        cal_previous_actions, args.device,
    )
    fits = {
        name: _fit_readout(cal_families[name], cal_target)
        for name in ("latent_static", "latent_temporal")
    }
    fits["rgb_geometry_action"] = _fit_readout(
        cal_color["color_geometry_action"], cal_target,
    )
    fits["rgb_geometry_temporal_action"] = _fit_readout(
        cal_color_temporal["color_geometry_temporal_action"], cal_target,
    )
    initial_conditions = [
        _capture_initial(args.seed + 1000 + index, int(args.tasks[index % len(args.tasks)]))
        for index in range(args.episodes)
    ]
    candidates = _candidate_bank()
    static = _run_arm(
        initial_conditions=initial_conditions, model=model, fit=fits["latent_static"],
        candidates=candidates, device=args.device, max_steps=args.max_steps,
        prelude_steps=args.prelude_steps, batch_size=args.batch_size, method="latent_static",
    )
    temporal = _run_arm(
        initial_conditions=initial_conditions, model=model, fit=fits["latent_temporal"],
        candidates=candidates, device=args.device, max_steps=args.max_steps,
        prelude_steps=args.prelude_steps, batch_size=args.batch_size, method="latent_temporal",
    )
    rgb_geometry = _run_arm(
        initial_conditions=initial_conditions, model=model,
        fit=fits["rgb_geometry_action"], candidates=candidates, device=args.device,
        max_steps=args.max_steps, prelude_steps=args.prelude_steps,
        batch_size=args.batch_size, method="rgb_geometry_action",
    )
    rgb_geometry_temporal = _run_arm(
        initial_conditions=initial_conditions, model=model,
        fit=fits["rgb_geometry_temporal_action"], candidates=candidates,
        device=args.device, max_steps=args.max_steps,
        prelude_steps=args.prelude_steps, batch_size=args.batch_size,
        method="rgb_geometry_temporal_action",
    )
    temporal_support_gate = _run_arm(
        initial_conditions=initial_conditions, model=model, fit=fits["latent_temporal"],
        candidates=candidates, device=args.device, max_steps=args.max_steps,
        prelude_steps=args.prelude_steps, batch_size=args.batch_size,
        method="latent_temporal", support_gate=True,
        support_threshold=args.support_threshold,
    )
    temporal_pixel_feedback = _run_arm(
        initial_conditions=initial_conditions, model=model, fit=fits["latent_temporal"],
        candidates=candidates, device=args.device, max_steps=args.max_steps,
        prelude_steps=args.prelude_steps, batch_size=args.batch_size,
        method="latent_temporal", pixel_feedback=True,
        pixel_feedback_threshold=args.pixel_feedback_threshold,
    )
    temporal_pixel_feedback_local = _run_arm(
        initial_conditions=initial_conditions, model=model, fit=fits["latent_temporal"],
        candidates=candidates, device=args.device, max_steps=args.max_steps,
        prelude_steps=args.prelude_steps, batch_size=args.batch_size,
        method="latent_temporal", pixel_feedback=True,
        pixel_feedback_threshold=args.pixel_feedback_threshold,
        pixel_feedback_mode="one_step",
    )
    temporal_rgb_rank_consensus = _run_arm(
        initial_conditions=initial_conditions, model=model,
        fit={
            "latent_temporal": fits["latent_temporal"],
            "rgb_geometry_action": fits["rgb_geometry_action"],
        }, candidates=candidates, device=args.device, max_steps=args.max_steps,
        prelude_steps=args.prelude_steps, batch_size=args.batch_size,
        method="latent_rgb_rank_consensus",
    )
    static_by_seed = {row["seed"]: row for row in static["episodes"]}
    temporal_by_seed = {row["seed"]: row for row in temporal["episodes"]}
    rgb_by_seed = {row["seed"]: row for row in rgb_geometry["episodes"]}
    rgb_temporal_by_seed = {
        row["seed"]: row for row in rgb_geometry_temporal["episodes"]
    }
    gate_by_seed = {row["seed"]: row for row in temporal_support_gate["episodes"]}
    pixel_feedback_by_seed = {row["seed"]: row for row in temporal_pixel_feedback["episodes"]}
    deltas = np.asarray([
        temporal_by_seed[seed]["final_goal_distance_m"] - static_by_seed[seed]["final_goal_distance_m"]
        for seed in static_by_seed
    ], dtype=np.float64)
    gate_deltas = np.asarray([
        gate_by_seed[seed]["final_goal_distance_m"] - static_by_seed[seed]["final_goal_distance_m"]
        for seed in static_by_seed
    ], dtype=np.float64)
    rgb_deltas = np.asarray([
        rgb_by_seed[seed]["final_goal_distance_m"] - static_by_seed[seed]["final_goal_distance_m"]
        for seed in static_by_seed
    ], dtype=np.float64)
    rgb_temporal_deltas = np.asarray([
        rgb_temporal_by_seed[seed]["final_goal_distance_m"]
        - static_by_seed[seed]["final_goal_distance_m"]
        for seed in static_by_seed
    ], dtype=np.float64)
    pixel_feedback_deltas = np.asarray([
        pixel_feedback_by_seed[seed]["final_goal_distance_m"]
        - static_by_seed[seed]["final_goal_distance_m"]
        for seed in static_by_seed
    ], dtype=np.float64)
    pixel_feedback_local_by_seed = {
        row["seed"]: row for row in temporal_pixel_feedback_local["episodes"]
    }
    consensus_by_seed = {
        row["seed"]: row for row in temporal_rgb_rank_consensus["episodes"]
    }
    pixel_feedback_local_deltas = np.asarray([
        pixel_feedback_local_by_seed[seed]["final_goal_distance_m"]
        - static_by_seed[seed]["final_goal_distance_m"]
        for seed in static_by_seed
    ], dtype=np.float64)
    consensus_deltas = np.asarray([
        consensus_by_seed[seed]["final_goal_distance_m"]
        - static_by_seed[seed]["final_goal_distance_m"]
        for seed in static_by_seed
    ], dtype=np.float64)
    report = {
        "schema": "cube_temporal_closed_loop.v1",
        "protocol": {
            "environment": "swm/OGBCube-v0",
            "checkpoint": args.policy,
            "checkpoint_frozen": True,
            "device": args.device,
            "calibration_roots": args.calibration_roots,
            "test_episodes": args.episodes,
            "root_disjoint": True,
            "same_initial_state_pairing": True,
            "runtime_input": "live current/previous RGB, rendered goal, previous action, fixed candidate bank",
            "runtime_policy_state_access": False,
            "fixed_prelude": "15 descent actions then 5 gripper-close actions",
            "support_gate": "if predicted candidate-score spread < fixed threshold, cycle x+/x-/y+/y-",
            "support_threshold_m": float(args.support_threshold),
            "pixel_feedback": "after the fixed prelude, reverse the next latent-temporal XY pulse when RGB block-goal error increases",
            "pixel_feedback_threshold": float(args.pixel_feedback_threshold),
            "pixel_feedback_mode": "persistent",
            "additional_feedback_arm": "one_step",
            "candidate_bank": candidates.tolist(),
            "readout_fit": "calibration roots only; no test outcomes",
            "evaluator_state_role": "initial-state pairing and final goal-distance scoring only",
        },
        "calibration": {"frames": int(len(cal_current)), "roots": int(args.calibration_roots)},
        "static": static,
        "temporal": temporal,
        "rgb_geometry_action": rgb_geometry,
        "rgb_geometry_temporal_action": rgb_geometry_temporal,
        "temporal_support_gate": temporal_support_gate,
        "temporal_pixel_feedback": temporal_pixel_feedback,
        "temporal_pixel_feedback_local": temporal_pixel_feedback_local,
        "temporal_rgb_rank_consensus": temporal_rgb_rank_consensus,
        "paired": {
            "seed_order": list(static_by_seed),
            "final_distance_delta_temporal_minus_static_m": deltas.tolist(),
            "mean_final_distance_delta_m": float(deltas.mean()),
            "temporal_wins": int(np.sum(deltas < 0.0)),
            "static_wins": int(np.sum(deltas > 0.0)),
            "ties": int(np.sum(deltas == 0.0)),
            "support_gate_final_distance_delta_vs_static_m": gate_deltas.tolist(),
            "support_gate_mean_final_distance_delta_m": float(gate_deltas.mean()),
            "support_gate_wins": int(np.sum(gate_deltas < 0.0)),
            "support_gate_losses": int(np.sum(gate_deltas > 0.0)),
            "support_gate_ties": int(np.sum(gate_deltas == 0.0)),
            "rgb_geometry_final_distance_delta_vs_static_m": rgb_deltas.tolist(),
            "rgb_geometry_mean_final_distance_delta_m": float(rgb_deltas.mean()),
            "rgb_geometry_wins": int(np.sum(rgb_deltas < 0.0)),
            "rgb_geometry_losses": int(np.sum(rgb_deltas > 0.0)),
            "rgb_geometry_ties": int(np.sum(rgb_deltas == 0.0)),
            "rgb_geometry_temporal_final_distance_delta_vs_static_m": rgb_temporal_deltas.tolist(),
            "rgb_geometry_temporal_mean_final_distance_delta_m": float(rgb_temporal_deltas.mean()),
            "rgb_geometry_temporal_wins": int(np.sum(rgb_temporal_deltas < 0.0)),
            "rgb_geometry_temporal_losses": int(np.sum(rgb_temporal_deltas > 0.0)),
            "rgb_geometry_temporal_ties": int(np.sum(rgb_temporal_deltas == 0.0)),
            "pixel_feedback_final_distance_delta_vs_static_m": pixel_feedback_deltas.tolist(),
            "pixel_feedback_mean_final_distance_delta_m": float(pixel_feedback_deltas.mean()),
            "pixel_feedback_wins": int(np.sum(pixel_feedback_deltas < 0.0)),
            "pixel_feedback_losses": int(np.sum(pixel_feedback_deltas > 0.0)),
            "pixel_feedback_ties": int(np.sum(pixel_feedback_deltas == 0.0)),
            "pixel_feedback_local_final_distance_delta_vs_static_m": pixel_feedback_local_deltas.tolist(),
            "pixel_feedback_local_mean_final_distance_delta_m": float(pixel_feedback_local_deltas.mean()),
            "pixel_feedback_local_wins": int(np.sum(pixel_feedback_local_deltas < 0.0)),
            "pixel_feedback_local_losses": int(np.sum(pixel_feedback_local_deltas > 0.0)),
            "pixel_feedback_local_ties": int(np.sum(pixel_feedback_local_deltas == 0.0)),
            "rank_consensus_final_distance_delta_vs_static_m": consensus_deltas.tolist(),
            "rank_consensus_mean_final_distance_delta_m": float(consensus_deltas.mean()),
            "rank_consensus_wins": int(np.sum(consensus_deltas < 0.0)),
            "rank_consensus_losses": int(np.sum(consensus_deltas > 0.0)),
            "rank_consensus_ties": int(np.sum(consensus_deltas == 0.0)),
        },
        "claim_boundary": (
            "Independent RGB-only closed-loop consequence for a fixed-prelude "
            "candidate ranker; it does not establish universal world-model "
            "transfer, causal latent dynamics, or stable pose closure."
        ),
    }
    (args.output / "closed_loop.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["paired"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
