"""Build a root-disjoint panel from the full online CEM proposal trajectory.

The initial Gaussian-support panel closes only the first support mismatch.  In
the actual controller, the frozen LeWM cost changes the proposal mean and
standard deviation after every elite update.  This runner mirrors the
standard_worldmodel ``CEMSolver`` update exactly (zero initial mean, unit
initial variance, first candidate forced to the current mean, top-k mean/std
update), records every unique proposal from all iterations, and labels those
proposals by low-level TwoRoom replay.

This is still an offline support/calibration artifact.  It does not use
outcome roots for fitting or selection and does not itself establish a live
planning gain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

import gymnasium as gym
import numpy as np
import torch

from experiments.audit_tworoom_shared_candidate_ranking import _Preprocessor, _model_rollout
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from experiments.run_live_tworoom_planning import (
    TwoRoomGoalAliasWrapper,
    _image_transform,
    _load_action_stats,
)
from experiments.gdd_wm.audit_tworoom_cem_support_panel import (
    _inverse_blocks,
    _rollout_raw_sequence,
    _split_for_root,
)


def _action_hash(action: np.ndarray) -> str:
    value = np.asarray(action, dtype=np.float32).round(7)
    value[value == 0.0] = 0.0
    return hashlib.sha256(value.tobytes()).hexdigest()[:20]


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.int64)
    result[order] = np.arange(len(values), dtype=np.int64)
    return result


def _support_hash(num_samples: int, n_steps: int, topk: int, horizon: int, action_block: int) -> str:
    payload = {
        "solver": "stable_worldmodel.CEMSolver",
        "num_samples": int(num_samples),
        "n_steps": int(n_steps),
        "topk": int(topk),
        "var_scale": 1.0,
        "initial_mean": 0.0,
        "horizon": int(horizon),
        "action_block": int(action_block),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]


def build_panel(
    *, policy: str, adapter_artifact: Path, output: Path, device: str,
    seed: int, roots: int, num_samples: int, n_steps: int, topk: int,
    horizon: int, action_block: int, history_len: int,
) -> dict[str, Any]:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    if roots < 5 or num_samples <= topk or n_steps < 1:
        raise ValueError("require roots>=5, num_samples>topk, and n_steps>=1")
    action_mean, action_std = _load_action_stats(adapter_artifact, "swm/TwoRoom-v1")
    width = int(action_block) * 2
    if action_mean.size != width or action_std.size != width:
        raise ValueError(
            f"action statistics width {action_mean.size} does not match action_block={action_block}"
        )
    model = load_compatible_pretrained(policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    preprocessor = _Preprocessor(ActionNormalizer(action_mean, action_std))
    env = TwoRoomGoalAliasWrapper(gym.make("swm/TwoRoom-v1", render_mode="rgb_array"))
    support_hash = _support_hash(num_samples, n_steps, topk, horizon, action_block)
    panel_rows: list[dict[str, Any]] = []
    predicted_latents: dict[str, np.ndarray] = {}
    realized_states: dict[str, np.ndarray] = {}
    root_summaries: list[dict[str, Any]] = []
    global_generator = torch.Generator(device=device).manual_seed(int(seed))
    try:
        for root_index in range(int(roots)):
            root_seed = int(seed + root_index)
            _, info = env.reset(seed=root_seed)
            current = np.asarray(env.render(), dtype=np.uint8).copy()
            goal = np.asarray(info["goal"], dtype=np.uint8).copy()
            batch_mean = torch.zeros((horizon, width), dtype=torch.float32, device=device)
            batch_var = torch.ones((horizon, width), dtype=torch.float32, device=device)
            unique: dict[str, dict[str, Any]] = {}
            iteration_sizes: list[int] = []
            source_latents_root: np.ndarray | None = None
            goal_latents_root: np.ndarray | None = None
            for iteration in range(int(n_steps)):
                candidates = torch.randn(
                    num_samples, horizon, width,
                    generator=global_generator,
                    device=device,
                    dtype=torch.float32,
                )
                candidates = candidates * batch_var.unsqueeze(0) + batch_mean.unsqueeze(0)
                candidates[0] = batch_mean
                model_cost, source_latents, predicted_root, goal_latents = _model_rollout(
                    model,
                    preprocessor,
                    current,
                    goal,
                    ActionNormalizer(action_mean, action_std),
                    candidates,
                    device,
                    history_len,
                    return_goal=True,
                )
                source_latents_root = source_latents
                goal_latents_root = goal_latents
                normalized = candidates.detach().cpu().numpy()
                raw_frames, clipped_frames = _inverse_blocks(
                    normalized,
                    action_mean,
                    action_std,
                    action_low=np.asarray(env.action_space.low, dtype=np.float32),
                    action_high=np.asarray(env.action_space.high, dtype=np.float32),
                )
                for candidate_index in range(num_samples):
                    action_key = _action_hash(clipped_frames[candidate_index])
                    if action_key not in unique:
                        unique[action_key] = {
                            "normalized": normalized[candidate_index].copy(),
                            "raw": raw_frames[candidate_index].copy(),
                            "clipped": clipped_frames[candidate_index].copy(),
                            "model_cost": float(model_cost[candidate_index]),
                            "predicted": np.asarray(predicted_root[candidate_index], dtype=np.float32).copy(),
                            "iteration": int(iteration),
                            "sample_index": int(candidate_index),
                        }
                topk_values, topk_indices = torch.topk(
                    torch.as_tensor(model_cost, device=device),
                    k=topk,
                    largest=False,
                )
                del topk_values
                topk_candidates = candidates[topk_indices]
                batch_mean = topk_candidates.mean(dim=0)
                batch_var = topk_candidates.std(dim=0)
                iteration_sizes.append(len(unique))
            if source_latents_root is None or goal_latents_root is None:
                raise RuntimeError("CEM produced no candidate support")

            entries = list(unique.values())
            true_cost = np.empty(len(entries), dtype=np.float64)
            success = np.zeros(len(entries), dtype=np.int8)
            clipping_fraction = np.empty(len(entries), dtype=np.float64)
            for index, entry in enumerate(entries):
                true_cost[index], terminal_success, terminal_state = _rollout_raw_sequence(
                    env, root_seed, entry["clipped"]
                )
                success[index] = int(terminal_success)
                clipping_fraction[index] = float(
                    np.mean(np.abs(entry["raw"] - entry["clipped"]) > 1.0e-7)
                )
                record_key = f"tworoom_episode_{root_seed}/{index:04d}"
                predicted_latents[record_key] = entry["predicted"]
                if terminal_state is not None:
                    realized_states[record_key] = terminal_state
            model_cost = np.asarray([float(entry["model_cost"]) for entry in entries], dtype=np.float64)
            model_rank = _rank(model_cost)
            true_rank = _rank(true_cost)
            predicted_best = int(np.argmin(model_cost))
            root_summaries.append({
                "seed": root_seed,
                "split": _split_for_root(root_index, int(roots)),
                "unique_candidate_count": len(entries),
                "iteration_unique_counts": iteration_sizes,
                "predicted_best": predicted_best,
                "true_best": int(np.argmin(true_cost)),
                "true_rank_of_predicted_best": int(true_rank[predicted_best] + 1),
                "regret": float(true_cost[predicted_best] - np.min(true_cost)),
                "success_count": int(success.sum()),
                "clipped_candidate_fraction": float(np.mean(clipping_fraction > 0.0)),
            })
            split = _split_for_root(root_index, int(roots))
            instance_id = f"tworoom_episode_{root_seed}"
            for index, entry in enumerate(entries):
                record_key = f"{instance_id}/{index:04d}"
                raw_sequence = entry["clipped"]
                raw_unclipped = entry["raw"]
                label = (
                    "cem_initial_mean" if entry["iteration"] == 0 and entry["sample_index"] == 0
                    else f"cem_iteration_{entry['iteration']}"
                )
                panel_rows.append({
                    "instance_id": instance_id,
                    "split": split,
                    "environment": "TwoRoom",
                    "start_episode_id": str(root_seed),
                    "goal_id": f"tworoom_goal_{root_seed}",
                    "candidate_id": f"candidate_{index:04d}",
                    "candidate_bank_hash": support_hash,
                    "action_sequence": raw_sequence.tolist(),
                    "normalized_action_sequence": entry["normalized"].tolist(),
                    "total_action_steps": int(raw_sequence.shape[0]),
                    "candidate_generator": label,
                    "cem_stage": label,
                    "cem_iteration": int(entry["iteration"]),
                    "source_observation_ref": f"live_rgb://TwoRoom/{root_seed}/current",
                    "goal_observation_ref": f"live_rgb://TwoRoom/{root_seed}/goal",
                    "source_state_raw": np.asarray(info["state"], dtype=np.float64).tolist(),
                    "goal_raw": np.asarray(info["goal_state"], dtype=np.float64).tolist(),
                    "teacher_model_id": "frozen_lewm_cost_v1",
                    "teacher_score": float(-entry["model_cost"]),
                    "teacher_rank": int(model_rank[index]),
                    "teacher_set_id": f"{instance_id}_cem_trajectory",
                    "real_success": bool(success[index]),
                    "real_task_cost": float(true_cost[index]),
                    "realized_observation_ref": f"offline_mujoco_state://{record_key}",
                    "predicted_latent_ref": f"arrays.npz::predicted::{record_key}",
                    "realized_latent_ref": None,
                    "realized_terminal_state_ref": f"arrays.npz::realized_state::{record_key}",
                    "action_support": {
                        "chart": "raw_control",
                        "proposal_distribution": "online_cem_trajectory",
                        "cem_iteration": int(entry["iteration"]),
                        "raw_min": float(np.min(raw_unclipped)),
                        "raw_max": float(np.max(raw_unclipped)),
                        "clipped_fraction": float(clipping_fraction[index]),
                        "duration_steps": int(raw_sequence.shape[0]),
                    },
                    "privileged_state_role": "offline_outcome_label_only",
                    "source_latent_model": np.asarray(source_latents_root[0], dtype=np.float64).tolist(),
                    "goal_latent_model": np.asarray(goal_latents_root[0], dtype=np.float64).tolist(),
                    "predicted_horizon": int(horizon),
                })
            print(
                f"[cem-trajectory] root={root_index + 1}/{roots} seed={root_seed} "
                f"split={split} unique={len(entries)} regret={root_summaries[-1]['regret']:.4f}",
                flush=True,
            )
    finally:
        env.close()

    output.mkdir(parents=True, exist_ok=True)
    arrays_path = output / "arrays.npz"
    np.savez_compressed(
        arrays_path,
        **{f"predicted::{key}": value for key, value in predicted_latents.items()},
        **{f"realized_state::{key}": value for key, value in realized_states.items()},
    )
    metadata = {
        "schema": "gdd_wm_tworoom_cem_trajectory_panel.v0.1",
        "environment": "TwoRoom",
        "candidate_bank_hash": support_hash,
        "candidate_bank_coordinate_frame": "normalized_cem_to_raw_replay",
        "candidate_count_per_root": sorted({
            int(item["unique_candidate_count"]) for item in root_summaries
        }),
        "candidate_budget_per_iteration": int(num_samples),
        "history_len": int(history_len),
        "horizon": int(horizon),
        "action_block": int(action_block),
        "replay_steps": int(horizon * action_block),
        "cem_solver": {
            "num_samples": int(num_samples),
            "n_steps": int(n_steps),
            "topk": int(topk),
            "var_scale": 1.0,
            "initial_mean": 0.0,
            "candidate_zero_mean_forced_first": True,
            "proposal_generator": "torch.Generator(device=cuda)",
            "unique_action_sequences_only_in_output": True,
        },
        "root_split": "first_half_train, second_half_evenly_split_calibration_outcome",
        "simulator_action_clip": "environment action-space bounds applied only during offline replay",
        "source_panel": "frozen RGB LeWM scored on full online CEM proposal trajectory",
        "panel_state_fields": "evaluator-side source_state_raw and goal_raw; never passed to model ranking",
        "runtime_teacher_forbidden": True,
        "outcome_used_for_teacher_or_metric_selection": False,
        "snapshot_selection_from_outcome": False,
        "teacher_frozen_before_distillation": True,
        "shared_candidate_bank": True,
        "checkpoint_frozen": True,
        "device": device,
        "realized_latent_status": "unavailable_without_RGB_rollout_for_new_simulated_sequences",
        "claim_boundary": (
            "offline full-CEM-support calibration panel; not closed-loop planning evidence "
            "until independent outcome execution"
        ),
    }
    dataset_path = output / "candidate_dataset.json"
    dataset_path.write_text(
        json.dumps({"metadata": metadata, "records": panel_rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "gdd_wm_tworoom_cem_trajectory_panel_manifest.v0.1",
        "candidate_dataset": str(dataset_path.resolve()),
        "arrays": str(arrays_path.resolve()),
        "record_count": len(panel_rows),
        "root_count": len(root_summaries),
        "split_root_counts": {
            split: len({row["instance_id"] for row in panel_rows if row["split"] == split})
            for split in ("train", "calibration", "outcome")
        },
        "root_summaries": root_summaries,
        "metadata": metadata,
        "claim_boundary": metadata["claim_boundary"],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="/home/gbk/.stable_worldmodel/checkpoints/tworoom/lewm")
    parser.add_argument(
        "--adapter-artifact", type=Path,
        default=Path("results/live_cross_environment_prediction_20261015/tworoom_cuda/cross_environment_prediction.json"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261090)
    parser.add_argument("--roots", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=8)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--history-len", type=int, default=1)
    args = parser.parse_args()
    manifest = build_panel(
        policy=args.policy,
        adapter_artifact=args.adapter_artifact,
        output=args.output,
        device=args.device,
        seed=args.seed,
        roots=args.roots,
        num_samples=args.num_samples,
        n_steps=args.n_steps,
        topk=args.topk,
        horizon=args.horizon,
        action_block=args.action_block,
        history_len=args.history_len,
    )
    print(json.dumps({
        "record_count": manifest["record_count"],
        "root_count": manifest["root_count"],
        "split_root_counts": manifest["split_root_counts"],
        "candidate_dataset": manifest["candidate_dataset"],
        "arrays": manifest["arrays"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
