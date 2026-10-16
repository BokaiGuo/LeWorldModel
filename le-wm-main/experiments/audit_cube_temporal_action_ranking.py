"""Shared-candidate action-ranking audit for Cube temporal latent readouts.

Calibration roots fit a frozen readout for the next block-to-goal delta.  On
held-out roots, an evaluator-only guided approach reaches a contact-support
state, then the exact state is branched across one shared action bank.  The
ranking input contains only current/previous RGB, rendered goal, and the
candidate action; privileged state is used solely to create the branch and to
compute oracle regret after the step.

This is intentionally a mechanism audit rather than RGB-only planning
evidence: guided approach coverage is not a deployable policy.  It tests the
necessary intermediate claim that the latent temporal readout can rank
within-state action effects before a closed-loop consequence is attempted.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import stable_worldmodel  # noqa: F401 - registers environments
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_cube_rgb_observability import (  # noqa: E402
    _exploratory_action,
    extract_color_geometry_features,
    extract_color_geometry_temporal_features,
    _goal_position,
    _make_env,
    _vec,
    fit_ridge,
)
from experiments.audit_cube_temporal_action_effect import (  # noqa: E402
    build_feature_families,
    collect_temporal_frames,
)
from experiments.checkpoint_compat import load_compatible_pretrained  # noqa: E402


def _candidate_bank(name: str = "legacy") -> np.ndarray:
    """Return a predeclared Cube action bank shared by every method/root.

    ``legacy`` is the historical protocol.  The post-contact lateral banks
    exclude vertical/gripper actions that are no longer controllable after
    contact.  They are explicit protocol interventions, not a test-outcome
    dependent selector.
    """
    if name == "legacy":
        actions = [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -1.0],
        ]
    elif name in {"lateral4", "lateral8"}:
        actions = [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0, 0.0],
        ]
        if name == "lateral8":
            diagonal = float(1.0 / np.sqrt(2.0))
            actions.extend([
                [diagonal, diagonal, 0.0, 0.0, 0.0],
                [diagonal, -diagonal, 0.0, 0.0, 0.0],
                [-diagonal, diagonal, 0.0, 0.0, 0.0],
                [-diagonal, -diagonal, 0.0, 0.0, 0.0],
            ])
    else:
        raise ValueError(f"unknown Cube candidate bank: {name}")
    return np.asarray(actions, dtype=np.float32)


def _oracle_mask(scores: np.ndarray, tolerance: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("oracle scores cannot be empty")
    return scores <= float(np.min(scores)) + float(tolerance)


def _capture_branch_point(
    *, seed: int, task_id: int, decision_frame: int,
) -> dict[str, Any]:
    if decision_frame < 0:
        raise ValueError("decision-frame must be non-negative")
    env = _make_env()
    try:
        obs, info = env.reset(
            seed=int(seed), options={"task_id": int(task_id), "render_goal": True},
        )
        goal_image = np.asarray(env.unwrapped._cur_goal_rendered, dtype=np.uint8).copy()
        guided_direction = (
            _vec(info, "privileged/block_0_pos", 3)[:2]
            - _vec(info, "proprio/effector_pos", 3)[:2]
        )
        previous_image = np.asarray(obs, dtype=np.uint8).copy()
        previous_action = np.zeros(5, dtype=np.float32)
        for frame_id in range(decision_frame):
            current = np.asarray(obs, dtype=np.uint8).copy()
            action = _exploratory_action(
                int(seed), frame_id, np.random.default_rng(seed + frame_id),
                direction=guided_direction,
            )
            next_obs, _, terminated, truncated, next_info = env.step(action)
            if terminated or truncated:
                raise RuntimeError("guided branch terminated before decision frame")
            previous_image = current
            previous_action = action.copy()
            obs, info = next_obs, next_info
        current_image = np.asarray(obs, dtype=np.uint8).copy()
        raw = env.unwrapped
        return {
            "seed": int(seed),
            "task_id": int(task_id),
            "variation_values": copy.deepcopy(raw.variation_space.value),
            "state": np.concatenate([raw.data.qpos.copy(), raw.data.qvel.copy()]).tolist(),
            "current_image": current_image,
            "previous_image": previous_image,
            "goal_image": goal_image,
            "previous_action": previous_action,
            "current_block": _vec(info, "privileged/block_0_pos", 3)[:2].copy(),
            "goal": _goal_position(env)[:2].copy(),
        }
    finally:
        env.close()


def _branch_observed_effects(
    branch: dict[str, Any], candidates: np.ndarray,
) -> np.ndarray:
    """Replay all candidates from one compiled, fixed branch state.

    Resetting/compiling MuJoCo once per candidate made the calibration-only
    support expansion unnecessarily expensive.  The state and variation are
    still fixed exactly; ``set_state`` plus ``pre_step``/``mj_forward`` only
    removes repeated model compilation and does not change the one-step label.
    """
    env = _make_env()
    try:
        env.reset(
            seed=int(branch["seed"]),
            options={
                "task_id": int(branch["task_id"]),
                "render_goal": True,
                "variation_values": copy.deepcopy(branch["variation_values"]),
                "state": np.asarray(branch["state"], dtype=np.float64),
            },
        )
        raw = env.unwrapped
        state = np.asarray(branch["state"], dtype=np.float64)
        qpos = state[: int(raw.model.nq)].copy()
        qvel = state[int(raw.model.nq):].copy()
        effects = []
        for action in candidates:
            raw.set_state(qpos, qvel)
            raw._reset_next_step = False
            raw.pre_step()
            mujoco.mj_forward(raw.model, raw.data)
            _, _, _, _, next_info = env.step(np.asarray(action, dtype=np.float32))
            next_block = _vec(next_info, "privileged/block_0_pos", 3)[:2]
            effects.append(np.asarray(branch["goal"], dtype=np.float32) - next_block)
        return np.asarray(effects, dtype=np.float32)
    finally:
        env.close()


def _predict(
    features: torch.Tensor, fit: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    mean, scale, weight = fit
    x = features.to(dtype=torch.float64)
    return ((x - mean) / scale.clamp_min(1.0e-4)) @ weight


def _fit_readout(
    features: torch.Tensor, target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    calibration = np.ones(features.shape[0], dtype=bool)
    _, mean, scale, weight = fit_ridge(features, target, calibration, ridge=1.0e-2)
    return mean, scale, weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", default="/home/gbk/.stable_worldmodel/checkpoints/cube/lewm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--calibration-roots", type=int, default=40)
    parser.add_argument("--held-out-roots", type=int, default=16)
    parser.add_argument("--decision-frame", type=int, default=20)
    parser.add_argument("--oracle-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for Cube action-ranking audit but unavailable")
    if args.calibration_roots < 8 or args.held_out_roots < 4:
        raise ValueError("need at least 8 calibration roots and 4 held-out roots")
    if args.oracle_tolerance < 0.0 or not np.isfinite(args.oracle_tolerance):
        raise ValueError("oracle-tolerance must be finite and non-negative")
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        "[purpose] Cube shared-candidate ranking: latent static versus latent temporal; "
        f"device={args.device} calibration_roots={args.calibration_roots} "
        f"held_out_roots={args.held_out_roots}"
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
    # This is a predeclared RGB-only geometry control.  It is deliberately
    # kept separate from the frozen latent families: if it succeeds where the
    # latent readouts fail, the negative result is an encoder/action-interface
    # boundary rather than a generic lack of visual information.
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
    calibration_features = {
        **cal_families,
        "rgb_geometry_action": cal_color["color_geometry_action"],
        "rgb_geometry_temporal_action": cal_color_temporal[
            "color_geometry_temporal_action"
        ],
    }
    candidates = _candidate_bank()
    records: list[dict[str, Any]] = []
    for index in range(args.held_out_roots):
        seed = args.seed + 1000 + index
        task_id = int(args.tasks[index % len(args.tasks)])
        branch = _capture_branch_point(
            seed=seed, task_id=task_id, decision_frame=args.decision_frame,
        )
        observed = _branch_observed_effects(branch, candidates)
        n_candidates = len(candidates)
        current = np.repeat(branch["current_image"][None], n_candidates, axis=0)
        previous = np.repeat(branch["previous_image"][None], n_candidates, axis=0)
        goals = np.repeat(branch["goal_image"][None], n_candidates, axis=0)
        previous_actions = np.repeat(branch["previous_action"][None], n_candidates, axis=0)
        valid = np.ones(n_candidates, dtype=np.float32)
        action = candidates.copy()
        families, _ = build_feature_families(
            current, previous, goals, action, previous_actions, valid,
            device=args.device, model=model, batch_size=args.batch_size,
        )
        color_families = extract_color_geometry_features(
            current, goals, action, args.device,
        )
        families["rgb_geometry_action"] = color_families["color_geometry_action"]
        families["rgb_geometry_temporal_action"] = extract_color_geometry_temporal_features(
            current, previous, goals, action, previous_actions, args.device,
        )["color_geometry_temporal_action"]
        actual_scores = np.linalg.norm(observed, axis=1)
        oracle = int(np.argmin(actual_scores))
        oracle_set = _oracle_mask(actual_scores, args.oracle_tolerance)
        row: dict[str, Any] = {
            "root_index": int(index),
            "seed": int(seed),
            "task_id": task_id,
            "decision_frame": int(args.decision_frame),
            "candidate_count": n_candidates,
            "oracle_index": oracle,
            "oracle_error_m": float(actual_scores[oracle]),
            "actual_candidate_error_m": actual_scores.tolist(),
        }
        for name in (
            "latent_static", "latent_temporal", "rgb_geometry_action",
            "rgb_geometry_temporal_action",
        ):
            predicted = _predict(families[name], fits[name]).detach().cpu().numpy()
            predicted_scores = np.linalg.norm(predicted, axis=1)
            selected = int(np.argmin(predicted_scores))
            row[name] = {
                "selected_index": selected,
                "selected_error_m": float(actual_scores[selected]),
                "predicted_candidate_error_m": predicted_scores.tolist(),
                "regret_m": float(actual_scores[selected] - actual_scores[oracle]),
                "oracle_agreement": bool(oracle_set[selected]),
            }
        records.append(row)

    report: dict[str, Any] = {
        "schema": "cube_temporal_action_ranking.v1",
        "protocol": {
            "environment": "swm/OGBCube-v0",
            "checkpoint": args.policy,
            "checkpoint_frozen": True,
            "device": args.device,
            "calibration_roots": args.calibration_roots,
            "held_out_roots": args.held_out_roots,
            "root_disjoint": True,
            "candidate_bank": candidates.tolist(),
            "candidate_bank_shared": True,
            "decision_input": "current/previous RGB, rendered goal, previous executed action, candidate action",
            "runtime_policy_state_access": False,
            "guided_branch_prep": True,
            "privileged_state_role": "branch construction and post-step oracle labels only",
            "readout_target": "next block-to-goal XY delta",
            "readout_fit": "calibration roots only; frozen before held-out ranking",
            "oracle_tolerance_m": float(args.oracle_tolerance),
        },
        "calibration": {
            "frames": int(len(cal_current)),
            "roots": int(args.calibration_roots),
            "feature_dims": {
                name: int(calibration_features[name].shape[1])
                for name in fits
            },
        },
        "held_out": {
            "roots": int(len(records)),
            "records": records,
        },
        "claim_boundary": (
            "Guided contact-support shared-candidate mechanism audit. It tests "
            "action ranking but is not pure RGB-only online planning evidence "
            "because branch preparation uses privileged geometry."
        ),
    }
    for name in (
        "latent_static", "latent_temporal", "rgb_geometry_action",
        "rgb_geometry_temporal_action",
    ):
        values = np.asarray([row[name]["regret_m"] for row in records], dtype=np.float64)
        report["held_out"][name] = {
            "mean_regret_m": float(values.mean()),
            "median_regret_m": float(np.median(values)),
            "oracle_agreement_rate": float(np.mean([row[name]["oracle_agreement"] for row in records])),
            "nonzero_regret_rate": float(np.mean(values > 0.0)),
        }
    (args.output / "ranking.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["held_out"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
