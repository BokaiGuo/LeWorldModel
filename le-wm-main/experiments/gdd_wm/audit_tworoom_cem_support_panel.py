"""Build a real CEM-support panel for the frozen RGB TwoRoom LeWM.

The earlier residual experiments calibrated on a fixed 17-action bank while
the live controller ranked Gaussian CEM proposals.  This runner removes that
support mismatch.  It samples the same *kind* of normalized action sequence
that the online solver sees (an initial zero/mean candidate plus Gaussian
proposals at frozen scales), scores those candidates with the frozen RGB model
on CUDA, and replays the inverse-normalized low-level actions in TwoRoom to
obtain offline labels.

The simulator state is used only for replay labels.  It is never passed to
the model scorer.  The output is intentionally an offline calibration panel;
it becomes a planning result only after a separate held-out live consequence
run.
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

from experiments.audit_tworoom_shared_candidate_ranking import (
    _Preprocessor,
    _model_rollout,
)
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from experiments.run_live_tworoom_planning import (
    TwoRoomGoalAliasWrapper,
    _image_transform,
    _load_action_stats,
)


def _split_for_root(index: int, total: int) -> str:
    """Deterministically create root-disjoint train/calibration/outcome splits."""

    if total < 5:
        raise ValueError("at least five roots are required for train/calibration/outcome")
    train_end = max(1, total // 2)
    calibration_end = train_end + max(1, (total - train_end) // 2)
    if index < train_end:
        return "train"
    if index < calibration_end:
        return "calibration"
    return "outcome"


def _parse_scales(value: str) -> tuple[float, ...]:
    scales = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not scales or any(scale <= 0.0 for scale in scales):
        raise ValueError("--scales must contain positive comma-separated values")
    return scales


def _support_hash(scales: tuple[float, ...], gaussian_per_scale: int, horizon: int, action_block: int) -> str:
    payload = {
        "scales": list(scales),
        "gaussian_per_scale": int(gaussian_per_scale),
        "horizon": int(horizon),
        "action_block": int(action_block),
        "mean_candidate": "normalized_zero",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]


def _sample_support(
    *, seed: int, scales: tuple[float, ...], gaussian_per_scale: int,
    horizon: int, action_block: int,
) -> tuple[np.ndarray, list[str]]:
    """Return normalized [N,H,5*2] candidates and their proposal labels."""

    if gaussian_per_scale < 1:
        raise ValueError("gaussian_per_scale must be positive")
    rng = np.random.default_rng(int(seed))
    width = int(action_block) * 2
    candidates = [np.zeros((horizon, width), dtype=np.float32)]
    labels = ["cem_mean_zero"]
    for scale in scales:
        draws = rng.normal(
            loc=0.0,
            scale=float(scale),
            size=(int(gaussian_per_scale), horizon, width),
        ).astype(np.float32)
        candidates.extend(draws)
        labels.extend([f"cem_gaussian_scale_{scale:g}"] * int(gaussian_per_scale))
    return np.stack(candidates, axis=0), labels


def _inverse_blocks(
    normalized: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray,
    *, action_low: np.ndarray, action_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert normalized model blocks to clipped low-level simulator actions."""

    normalized = np.asarray(normalized, dtype=np.float32)
    if normalized.ndim != 3 or normalized.shape[-1] != action_mean.size:
        raise ValueError(
            "normalized candidates must have shape [N,H,width] matching action statistics"
        )
    raw_blocks = normalized * action_std.reshape(1, 1, -1) + action_mean.reshape(1, 1, -1)
    frame_count = raw_blocks.shape[-1] // 2
    raw_frames = raw_blocks.reshape(raw_blocks.shape[0], raw_blocks.shape[1] * frame_count, 2)
    low = np.asarray(action_low, dtype=np.float32).reshape(1, 1, 2)
    high = np.asarray(action_high, dtype=np.float32).reshape(1, 1, 2)
    clipped_frames = np.clip(raw_frames, low, high)
    return raw_frames, clipped_frames


def _rollout_raw_sequence(
    env: gym.Env, seed: int, sequence: np.ndarray,
) -> tuple[float, bool, np.ndarray | None]:
    """Replay one candidate from the exact root seed using low-level actions."""

    env.reset(seed=int(seed))
    final_info: dict[str, Any] = {}
    terminated = truncated = False
    for action in np.asarray(sequence, dtype=np.float32):
        _, _, terminated, truncated, final_info = env.step(action)
        if terminated or truncated:
            break
    distance = float(np.asarray(final_info.get("distance_to_target", np.inf)).reshape(-1)[0])
    state = final_info.get("state")
    state_array = None if state is None else np.asarray(state, dtype=np.float32).copy()
    return distance, bool(terminated), state_array


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.int64)
    result[order] = np.arange(len(values), dtype=np.int64)
    return result


def build_panel(
    *, policy: str, adapter_artifact: Path, output: Path, device: str,
    seed: int, roots: int, gaussian_per_scale: int, scales: tuple[float, ...],
    horizon: int, action_block: int, history_len: int,
) -> dict[str, Any]:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    if horizon < 1 or action_block < 1 or history_len < 1:
        raise ValueError("horizon, action_block, and history_len must be positive")

    action_mean, action_std = _load_action_stats(adapter_artifact, "swm/TwoRoom-v1")
    expected_width = int(horizon * 0 + action_block * 2)
    if action_mean.size != expected_width or action_std.size != expected_width:
        raise ValueError(
            "TwoRoom action statistics do not match action_block: "
            f"stats={action_mean.size}, expected={expected_width}"
        )
    model = load_compatible_pretrained(policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    preprocessor = _Preprocessor(ActionNormalizer(action_mean, action_std))
    env = TwoRoomGoalAliasWrapper(gym.make("swm/TwoRoom-v1", render_mode="rgb_array"))
    support_hash = _support_hash(scales, gaussian_per_scale, horizon, action_block)
    panel_rows: list[dict[str, Any]] = []
    predicted_latents: dict[str, np.ndarray] = {}
    realized_states: dict[str, np.ndarray] = {}
    root_summaries: list[dict[str, Any]] = []
    total = int(roots)
    try:
        for root_index in range(total):
            root_seed = int(seed + root_index)
            _, info = env.reset(seed=root_seed)
            current = np.asarray(env.render(), dtype=np.uint8).copy()
            goal = np.asarray(info["goal"], dtype=np.uint8).copy()
            normalized, proposal_labels = _sample_support(
                seed=root_seed + 7919,
                scales=scales,
                gaussian_per_scale=gaussian_per_scale,
                horizon=horizon,
                action_block=action_block,
            )
            candidates = torch.as_tensor(normalized, dtype=torch.float32)
            model_cost, source_latents, predicted_latents_root, goal_latents = _model_rollout(
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
            raw_frames, clipped_frames = _inverse_blocks(
                normalized,
                action_mean,
                action_std,
                action_low=np.asarray(env.action_space.low, dtype=np.float32),
                action_high=np.asarray(env.action_space.high, dtype=np.float32),
            )
            true_cost = np.empty(len(normalized), dtype=np.float64)
            successes = np.zeros(len(normalized), dtype=np.int8)
            clipping_fraction = np.mean(np.abs(raw_frames - clipped_frames) > 1.0e-7, axis=(1, 2))
            for candidate_index in range(len(normalized)):
                cost, success, terminal_state = _rollout_raw_sequence(
                    env, root_seed, clipped_frames[candidate_index]
                )
                true_cost[candidate_index] = cost
                successes[candidate_index] = int(success)
                record_key = f"tworoom_episode_{root_seed}/{candidate_index:03d}"
                predicted_latents[record_key] = np.asarray(
                    predicted_latents_root[candidate_index], dtype=np.float32
                )
                if terminal_state is not None:
                    realized_states[record_key] = terminal_state

            model_rank = _rank(model_cost)
            true_rank = _rank(true_cost)
            predicted_best = int(np.argmin(model_cost))
            root_summaries.append({
                "seed": root_seed,
                "split": _split_for_root(root_index, total),
                "predicted_best": predicted_best,
                "true_best": int(np.argmin(true_cost)),
                "true_rank_of_predicted_best": int(true_rank[predicted_best] + 1),
                "regret": float(true_cost[predicted_best] - np.min(true_cost)),
                "model_cost_min": float(np.min(model_cost)),
                "true_cost_min": float(np.min(true_cost)),
                "success_count": int(successes.sum()),
                "clipped_candidate_fraction": float(np.mean(clipping_fraction > 0.0)),
            })
            split = _split_for_root(root_index, total)
            instance_id = f"tworoom_episode_{root_seed}"
            for candidate_index in range(len(normalized)):
                record_key = f"{instance_id}/{candidate_index:03d}"
                raw_sequence = clipped_frames[candidate_index]
                raw_unclipped = raw_frames[candidate_index]
                panel_rows.append({
                    "instance_id": instance_id,
                    "split": split,
                    "environment": "TwoRoom",
                    "start_episode_id": str(root_seed),
                    "goal_id": f"tworoom_goal_{root_seed}",
                    "candidate_id": f"candidate_{candidate_index:03d}",
                    "candidate_bank_hash": support_hash,
                    "action_sequence": raw_sequence.tolist(),
                    "normalized_action_sequence": normalized[candidate_index].tolist(),
                    "total_action_steps": int(raw_sequence.shape[0]),
                    "candidate_generator": proposal_labels[candidate_index],
                    "cem_stage": "initial_gaussian_proposal" if candidate_index else "cem_initial_mean",
                    "source_observation_ref": f"live_rgb://TwoRoom/{root_seed}/current",
                    "goal_observation_ref": f"live_rgb://TwoRoom/{root_seed}/goal",
                    "source_state_raw": np.asarray(info["state"], dtype=np.float64).tolist(),
                    "goal_raw": np.asarray(info["goal_state"], dtype=np.float64).tolist(),
                    "teacher_model_id": "frozen_lewm_cost_v1",
                    "teacher_score": float(-model_cost[candidate_index]),
                    "teacher_rank": int(model_rank[candidate_index]),
                    "teacher_set_id": f"{instance_id}_cem_support",
                    "real_success": bool(successes[candidate_index]),
                    "real_task_cost": float(true_cost[candidate_index]),
                    "realized_observation_ref": f"offline_mujoco_state://{record_key}",
                    "predicted_latent_ref": f"arrays.npz::predicted::{record_key}",
                    "realized_latent_ref": None,
                    "realized_terminal_state_ref": f"arrays.npz::realized_state::{record_key}",
                    "action_support": {
                        "chart": "raw_control",
                        "proposal_distribution": "normalized_gaussian",
                        "proposal_scale": 0.0 if candidate_index == 0 else float(proposal_labels[candidate_index].rsplit("_", 1)[-1]),
                        "normalized_mean": 0.0,
                        "normalized_std": float(np.std(normalized[candidate_index])),
                        "raw_min": float(np.min(raw_unclipped)),
                        "raw_max": float(np.max(raw_unclipped)),
                        "clipped_fraction": float(clipping_fraction[candidate_index]),
                        "duration_steps": int(raw_sequence.shape[0]),
                    },
                    "privileged_state_role": "offline_outcome_label_only",
                    "source_latent_model": np.asarray(source_latents[candidate_index], dtype=np.float64).tolist(),
                    "goal_latent_model": np.asarray(goal_latents[candidate_index], dtype=np.float64).tolist(),
                    "predicted_horizon": int(horizon),
                })
            print(
                f"[cem-support] root={root_index + 1}/{total} seed={root_seed} "
                f"split={split} model_regret={root_summaries[-1]['regret']:.4f} "
                f"clip={root_summaries[-1]['clipped_candidate_fraction']:.3f}",
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
        "schema": "gdd_wm_tworoom_cem_support_panel.v0.1",
        "environment": "TwoRoom",
        "candidate_bank_hash": support_hash,
        "candidate_bank_coordinate_frame": "normalized_gaussian_to_raw_replay",
        "candidate_count": int(1 + len(scales) * gaussian_per_scale),
        "history_len": int(history_len),
        "horizon": int(horizon),
        "action_block": int(action_block),
        "replay_steps": int(horizon * action_block),
        "proposal_distribution": {
            "mean_candidate": "all_zero_normalized_action",
            "gaussian_scales": list(scales),
            "gaussian_candidates_per_scale": int(gaussian_per_scale),
            "normalized_action_dim": int(action_mean.size),
        },
        "root_split": "first_half_train, second_half_evenly_split_calibration_outcome",
        "simulator_action_clip": "environment action-space bounds applied only during offline replay",
        "source_panel": "frozen RGB LeWM scored on exact CEM-style Gaussian support",
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
            "offline CEM-support calibration panel; not closed-loop planning evidence "
            "until independent outcome execution"
        ),
    }
    dataset_path = output / "candidate_dataset.json"
    dataset_path.write_text(
        json.dumps({"metadata": metadata, "records": panel_rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "gdd_wm_tworoom_cem_support_panel_manifest.v0.1",
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
    parser.add_argument("--seed", type=int, default=20261080)
    parser.add_argument("--roots", type=int, default=32)
    parser.add_argument("--gaussian-per-scale", type=int, default=21)
    parser.add_argument("--scales", default="1.0,0.5,0.25")
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
        gaussian_per_scale=args.gaussian_per_scale,
        scales=_parse_scales(args.scales),
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
