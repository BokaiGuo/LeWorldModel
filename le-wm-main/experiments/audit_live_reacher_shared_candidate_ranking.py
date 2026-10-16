"""Root-disjoint RGB action-ranking audit for the frozen Reacher LeWM.

The model ranks a fixed macro-action bank from rendered current/goal RGB.  The
same candidates are replayed from the same reset seed to obtain an
evaluator-only terminal qpos error.  Simulator state is never passed to the
model and is not used to select the checkpoint, candidate bank, or calibration
split.  This is a task-transfer diagnostic, not a claim of universal causal
world modelling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# Preserve the repository's historical direct-script entry point in addition
# to ``python -m experiments...``.
if __package__ in (None, ""):
    import sys

    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm
import torch
from stable_worldmodel.policy import BasePolicy

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from experiments.run_live_reacher_planning import (
    ReacherGoalImageWrapper,
    _image_transform,
    _load_action_stats,
    install_missing_field_compat,
)


def _directions() -> list[tuple[float, float]]:
    values = [(0.0, 0.0)]
    for magnitude in (0.25, 0.50, 0.75):
        values.extend(
            [
                (magnitude, 0.0),
                (-magnitude, 0.0),
                (0.0, magnitude),
                (0.0, -magnitude),
            ]
        )
    values.extend([(0.5, 0.5), (0.5, -0.5), (-0.5, 0.5), (-0.5, -0.5)])
    return values


def _candidate_bank(
    mean: np.ndarray,
    std: np.ndarray,
    *,
    horizon: int = 5,
    action_block: int = 5,
) -> tuple[np.ndarray, torch.Tensor]:
    raw = np.asarray(_directions(), dtype=np.float32)
    normalized = []
    for frame in raw:
        block = np.tile(frame, (action_block, 1)).reshape(-1)
        normalized_block = (block - mean) / np.maximum(std, 1e-6)
        normalized.append(np.tile(normalized_block[None, :], (horizon, 1)))
    return raw, torch.as_tensor(np.stack(normalized), dtype=torch.float32)


class _Preprocessor(BasePolicy):
    def __init__(self, action_normalizer: ActionNormalizer):
        super().__init__()
        self.process = {"action": action_normalizer}
        self.transform = {"pixels": _image_transform(), "goal": _image_transform()}


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    a = _rankdata(left)
    b = _rankdata(right)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _wrap_angle(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _target_qpos(env: gym.Env) -> np.ndarray:
    raw = env.unwrapped
    target = getattr(raw.env.task, "target_qpos", None)
    if target is None:
        raise RuntimeError("Reacher task did not expose target_qpos to the evaluator")
    return np.asarray(target, dtype=np.float64).copy()


def _current_qpos(env: gym.Env) -> np.ndarray:
    return np.asarray(env.unwrapped.env.physics.data.qpos, dtype=np.float64).copy()


def _rollout_candidate(
    env: gym.Env,
    seed: int,
    action: np.ndarray,
    steps: int,
    action_block: int,
) -> float:
    env.reset(seed=int(seed))
    target = _target_qpos(env)
    sequence = np.asarray(action, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[1] != 2:
        raise ValueError(f"candidate action must have shape [T,2], got {sequence.shape}")
    for step in range(steps):
        macro_index = min(step // max(int(action_block), 1), len(sequence) - 1)
        _, _, terminated, truncated, _ = env.step(sequence[macro_index])
        if terminated or truncated:
            break
    error = _wrap_angle(_current_qpos(env)[:2] - target[:2])
    return float(np.linalg.norm(error))


@torch.inference_mode()
def _model_rollout(
    model: torch.nn.Module,
    preprocessor: _Preprocessor,
    current: np.ndarray,
    goal: np.ndarray,
    candidates: torch.Tensor,
    device: str,
    history_len: int,
    *,
    return_goal: bool = False,
) -> np.ndarray:
    if history_len <= 0:
        raise ValueError("history_len must be positive")
    raw = {
        "pixels": np.repeat(current[None, None], history_len, axis=1),
        "goal": goal[None, None],
        "action": np.zeros((1, history_len, 2), dtype=np.float32),
    }
    prepared = preprocessor._prepare_info(raw)
    bank_size = int(candidates.shape[0])
    model_info = {}
    for key in ("pixels", "goal", "action"):
        value = prepared[key].to(device)
        model_info[key] = value[:, None].expand(1, bank_size, *value.shape[1:]).contiguous()
    candidate_tensor = candidates.to(device).unsqueeze(0)
    cost = model.get_cost(model_info, candidate_tensor).squeeze(0).detach().cpu().numpy()
    source = model_info["emb"][0, :, -1].detach().cpu().numpy()
    predicted = model_info["predicted_emb"][0, :, -1].detach().cpu().numpy()
    goal_emb = model_info["goal_emb"]
    if goal_emb.ndim == 4:
        goal = goal_emb[0, :, -1].detach().cpu().numpy()
    elif goal_emb.ndim == 3:
        goal = goal_emb[0, -1].detach().cpu().numpy()[None, :]
        goal = np.repeat(goal, bank_size, axis=0)
    else:
        raise RuntimeError(f"unexpected goal embedding shape: {tuple(goal_emb.shape)}")
    result = (cost, source, predicted)
    return result + (goal,) if return_goal else result


def _model_costs(
    model: torch.nn.Module,
    preprocessor: _Preprocessor,
    current: np.ndarray,
    goal: np.ndarray,
    candidates: torch.Tensor,
    device: str,
    history_len: int,
) -> np.ndarray:
    """Backward-compatible scalar-cost wrapper for existing ranking audits."""
    return _model_rollout(model, preprocessor, current, goal, candidates, device, history_len)[0]


def _split_for_root(index: int, total: int) -> str:
    train_end = max(1, total // 2)
    calibration_end = train_end + max(1, (total - train_end) // 2)
    if index < train_end:
        return "train"
    if index < calibration_end:
        return "calibration"
    return "outcome"


def _write_candidate_panel(
    output: Path,
    *,
    args: argparse.Namespace,
    raw_bank: np.ndarray,
    panel_rows: list[dict],
    predicted_latents: dict[str, np.ndarray],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    arrays_path = output / "arrays.npz"
    np.savez_compressed(
        arrays_path,
        **{f"predicted::{key}": value for key, value in predicted_latents.items()},
    )
    bank_hash = hashlib.sha256(
        np.asarray(raw_bank, dtype=np.float64).round(8).tobytes()
    ).hexdigest()[:20]
    for row in panel_rows:
        row["candidate_bank_hash"] = bank_hash
    metadata = {
        "schema": "gdd_wm_live_candidate_panel.v0.2",
        "environment": "Reacher",
        "candidate_bank_hash": bank_hash,
        "candidate_bank_coordinate_frame": "raw_control",
        "candidate_bank_count": int(len(raw_bank)),
        "history_len": int(args.history_len),
        "replay_steps": int(args.replay_steps),
        "target_delta": [float(value) for value in args.target_delta],
        "root_split": "first_half_train, second_half_evenly_split_calibration_outcome",
        "source_panel": "frozen RGB LeWM candidate ranking with simulator labels only",
        "panel_state_fields": "evaluator-side source_state_raw and goal_raw; never passed to model ranking",
        "runtime_teacher_forbidden": True,
        "outcome_used_for_teacher_or_metric_selection": False,
        "claim_boundary": (
            "Reacher offline candidate panel for train-only latent readout diagnostics; "
            "not D-JEPA teacher, causal physics, or closed-loop planning evidence."
        ),
    }
    dataset_path = output / "candidate_dataset.json"
    dataset_path.write_text(
        json.dumps({"metadata": metadata, "records": panel_rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "gdd_wm_live_candidate_panel_manifest.v0.2",
        "candidate_dataset": str(dataset_path.resolve()),
        "arrays": str(arrays_path.resolve()),
        "record_count": len(panel_rows),
        "root_count": len({row["instance_id"] for row in panel_rows}),
        "split_root_counts": {
            split: len({row["instance_id"] for row in panel_rows if row["split"] == split})
            for split in ("train", "calibration", "outcome")
        },
        "claim_boundary": metadata["claim_boundary"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pairwise_inversion(predicted: np.ndarray, realized: np.ndarray) -> float:
    total = 0
    inverted = 0
    for left in range(len(predicted)):
        for right in range(left + 1, len(predicted)):
            if np.isclose(realized[left], realized[right]):
                continue
            total += 1
            if (predicted[left] - predicted[right]) * (realized[left] - realized[right]) < 0:
                inverted += 1
    return float(inverted / total) if total else 0.0


def _bootstrap_mean(values: np.ndarray, *, seed: int, samples: int = 4000) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return [float(values.mean()), float(values.mean())]
    generator = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = generator.choice(values, size=len(values), replace=True).mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main() -> None:
    install_missing_field_compat()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="/home/gbk/.stable_worldmodel/checkpoints/reacher/lewm")
    parser.add_argument(
        "--adapter-artifact",
        type=Path,
        default=Path("results/live_cross_environment_prediction_20261015/reacher_cuda/cross_environment_prediction.json"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261320)
    parser.add_argument("--roots", type=int, default=32)
    parser.add_argument("--replay-steps", type=int, default=25)
    parser.add_argument(
        "--history-len",
        type=int,
        default=3,
        help="Deployment-matched RGB history length; Reacher live policy uses 3.",
    )
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument(
        "--target-delta",
        type=float,
        nargs=2,
        default=(0.65, -0.65),
        metavar=("DELTA_Q1", "DELTA_Q2"),
        help="Predeclared evaluator-side target-qpos offset.",
    )
    parser.add_argument(
        "--panel-output", type=Path, default=None,
        help="optional directory for source/predicted latent plus outcome candidate panel",
    )
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.roots < 2 or args.replay_steps < 1:
        raise ValueError("roots must be >=2 and replay-steps must be positive")

    action_mean, action_std = _load_action_stats(
        args.adapter_artifact, "swm/ReacherDMControl-v0"
    )
    normalizer = ActionNormalizer(action_mean, action_std)
    raw_bank, normalized_bank = _candidate_bank(
        action_mean, action_std, horizon=args.horizon, action_block=args.action_block
    )
    model = load_compatible_pretrained(args.policy, device=args.device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    preprocessor = _Preprocessor(normalizer)
    env = ReacherGoalImageWrapper(
        gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"),
        target_delta=args.target_delta,
    )
    rows: list[dict] = []
    panel_rows: list[dict] = []
    predicted_latents: dict[str, np.ndarray] = {}
    try:
        for root_index in range(args.roots):
            seed = int(args.seed + root_index)
            _, info = env.reset(seed=seed)
            current = np.asarray(env.render(), dtype=np.uint8).copy()
            goal = np.asarray(info["goal"], dtype=np.uint8).copy()
            source_state_raw = _current_qpos(env)
            goal_raw = _target_qpos(env)
            model_cost, source_latents, predicted_terminal_latents, goal_latents = _model_rollout(
                model, preprocessor, current, goal, normalized_bank, args.device, args.history_len,
                return_goal=True,
            )
            realized_cost = np.asarray(
                [
                    _rollout_candidate(
                        env,
                        seed,
                        np.tile(action[None, :], (args.action_block, 1)),
                        args.replay_steps,
                        args.action_block,
                    )
                    for action in raw_bank
                ],
                dtype=np.float64,
            )
            predicted_best = int(np.argmin(model_cost))
            realized_order = np.argsort(realized_cost, kind="stable")
            rows.append(
                {
                    "seed": seed,
                    "predicted_best": predicted_best,
                    "realized_best": int(realized_order[0]),
                    "realized_rank_of_predicted_best": int(
                        np.flatnonzero(realized_order == predicted_best)[0] + 1
                    ),
                    "top1_hit": int(predicted_best == int(realized_order[0])),
                    "top4_hit": int(predicted_best in set(realized_order[:4].tolist())),
                    "action_ranking_inversion": _pairwise_inversion(model_cost, realized_cost),
                    "multistep_task_error": float(realized_cost[predicted_best]),
                    "planning_regret": float(
                        realized_cost[predicted_best] - realized_cost[realized_order[0]]
                    ),
                    "angle_closure": float(realized_cost[predicted_best]),
                    "spearman": _spearman(model_cost, realized_cost),
                    "model_cost": model_cost.tolist(),
                    "realized_cost": realized_cost.tolist(),
                }
            )
            if args.panel_output is not None:
                split = _split_for_root(root_index, args.roots)
                instance_id = f"reacher_episode_{seed}"
                for candidate_index, action in enumerate(raw_bank):
                    record_key = f"{instance_id}/{candidate_index:03d}"
                    sequence = np.tile(
                        np.asarray(action, dtype=np.float32)[None, :],
                        (args.replay_steps, 1),
                    )
                    predicted_latents[record_key] = np.asarray(
                        predicted_terminal_latents[candidate_index], dtype=np.float32
                    )
                    panel_rows.append({
                        "instance_id": instance_id,
                        "split": split,
                        "environment": "Reacher",
                        "start_episode_id": str(seed),
                        "candidate_id": f"candidate_{candidate_index:03d}",
                        "candidate_bank_hash": "pending_written_by_panel_writer",
                        "action_sequence": sequence.tolist(),
                        "total_action_steps": int(args.replay_steps),
                        "candidate_generator": "reacher_shared_directions_v1",
                        "cem_stage": "fixed_counterfactual_bank",
                        "source_observation_ref": f"live_rgb://Reacher/{seed}/current",
                        "goal_observation_ref": f"live_rgb://Reacher/{seed}/goal",
                        "source_state_raw": source_state_raw.tolist(),
                        "goal_raw": goal_raw.tolist(),
                        "teacher_model_id": "frozen_lewm_cost_v1",
                        "teacher_score": float(-model_cost[candidate_index]),
                        "teacher_rank": int(np.flatnonzero(np.argsort(model_cost) == candidate_index)[0]),
                        "teacher_set_id": f"{instance_id}_shared_bank",
                        "real_success": None,
                        "real_task_cost": float(realized_cost[candidate_index]),
                        "realized_observation_ref": f"offline_dmcontrol_state://{record_key}",
                        "predicted_latent_ref": f"arrays.npz::predicted::{record_key}",
                        "realized_latent_ref": None,
                        "action_support": {
                            "chart": "raw_control",
                            "min": float(np.min(sequence)),
                            "max": float(np.max(sequence)),
                            "duration_steps": int(args.replay_steps),
                        },
                        "privileged_state_role": "offline_outcome_label_only",
                        "source_latent_model": np.asarray(
                            source_latents[candidate_index], dtype=np.float64
                        ).tolist(),
                        "goal_latent_model": np.asarray(
                            goal_latents[candidate_index], dtype=np.float64
                        ).tolist(),
                        "predicted_horizon": int(args.horizon),
                    })
    finally:
        env.close()

    if args.panel_output is not None:
        _write_candidate_panel(
            args.panel_output,
            args=args,
            raw_bank=raw_bank,
            panel_rows=panel_rows,
            predicted_latents=predicted_latents,
        )

    inversion = np.asarray([row["action_ranking_inversion"] for row in rows], dtype=np.float64)
    task_error = np.asarray([row["multistep_task_error"] for row in rows], dtype=np.float64)
    angle_error = np.asarray([row["angle_closure"] for row in rows], dtype=np.float64)
    regret = np.asarray([row["planning_regret"] for row in rows], dtype=np.float64)
    payload = {
        "schema": "reacher_shared_candidate_ranking.v1",
        "protocol": {
            "roots": int(args.roots),
            "root_disjoint": True,
            "candidate_count": int(len(raw_bank)),
            "candidate_bank": "zero, axis-aligned magnitudes 0.25/0.50/0.75, and four diagonals",
            "replay_steps": int(args.replay_steps),
            "action_block": int(args.action_block),
            "history_len": int(args.history_len),
            "horizon": int(args.horizon),
            "model_input": "RGB current frame, RGB rendered target frame, and action/history only",
            "runtime_state_access": False,
            "simulator_use": "held-out labels only; never passed to model ranking",
            "checkpoint_frozen": True,
            "test_planning_used_for_selection": False,
            "target_delta": [float(value) for value in args.target_delta],
            "device": args.device,
            "panel_output": str(args.panel_output.resolve()) if args.panel_output is not None else None,
        },
        "roots": rows,
        "metrics": {
            "action_ranking_inversion": {
                "status": "available",
                "estimate": float(inversion.mean()),
                "ci95": _bootstrap_mean(inversion, seed=args.seed + 11),
                "direction": "lower_is_better",
            },
            "multistep_task_error": {
                "status": "available",
                "estimate": float(task_error.mean()),
                "ci95": _bootstrap_mean(task_error, seed=args.seed + 12),
                "direction": "lower_is_better",
                "definition": "realized terminal qpos error after model-selected macro action",
            },
            "angle_closure": {
                "status": "available",
                "estimate": float(angle_error.mean()),
                "ci95": _bootstrap_mean(angle_error, seed=args.seed + 13),
                "direction": "lower_is_better",
                "definition": "wrapped two-joint terminal qpos error",
            },
            "planning_regret": {
                "status": "available",
                "estimate": float(regret.mean()),
                "ci95": _bootstrap_mean(regret, seed=args.seed + 14),
                "direction": "lower_is_better",
                "definition": "selected-candidate terminal error minus oracle-bank terminal error",
            },
            "contact_retention": {
                "status": "not_applicable",
                "reason": "Reacher is a non-contact qpos-match task",
            },
        },
        "summary": {
            "n": int(len(rows)),
            "top1_accuracy": float(np.mean([row["top1_hit"] for row in rows])),
            "top4_accuracy": float(np.mean([row["top4_hit"] for row in rows])),
            "mean_spearman": float(np.mean([row["spearman"] for row in rows])),
        },
        "claim_boundary": (
            "Root-disjoint Reacher RGB action-ranking and terminal task-error diagnostic; "
            "not a contact-retention result, universal world model, or transfer certificate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": payload["summary"], "metrics": payload["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
