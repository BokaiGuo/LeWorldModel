"""Build a train/calibration/outcome panel matching the RGB door bank.

The candidate bank is generated only from the current and goal RGB frames,
using the exact ``vision_door16`` helper used by live planning.  Simulator
state is used only for replay labels.  This closes the support mismatch before
fitting a task-conditioned teacher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from experiments.audit_tworoom_shared_candidate_ranking import _Preprocessor, _model_rollout
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.gdd_wm.audit_tworoom_cem_support_panel import _inverse_blocks, _rollout_raw_sequence, _split_for_root
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from experiments.run_live_tworoom_planning import (
    TwoRoomGoalAliasWrapper,
    _image_transform,
    _load_action_stats,
    _vision_door_candidate_bank,
)


def _hash(horizon: int, action_block: int) -> str:
    text = json.dumps({
        "variant": "vision_door16",
        "horizon": int(horizon),
        "action_block": int(action_block),
        "offsets": 8,
        "magnitudes": [0.5, 0.75],
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def build(*, policy: str, adapter_artifact: Path, output: Path, device: str,
          seed: int, roots: int, horizon: int, action_block: int,
          history_len: int) -> dict[str, Any]:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    action_mean, action_std = _load_action_stats(adapter_artifact, "swm/TwoRoom-v1")
    if action_mean.size != action_block * 2:
        raise ValueError("action statistics do not match action_block")
    model = load_compatible_pretrained(policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    preprocessor = _Preprocessor(ActionNormalizer(action_mean, action_std))
    env = TwoRoomGoalAliasWrapper(gym.make("swm/TwoRoom-v1", render_mode="rgb_array"))
    rows: list[dict[str, Any]] = []
    predicted_latents: dict[str, np.ndarray] = {}
    realized_states: dict[str, np.ndarray] = {}
    bank_hash = _hash(horizon, action_block)
    try:
        for root_index in range(int(roots)):
            root_seed = int(seed + root_index)
            _, info = env.reset(seed=root_seed)
            current = np.asarray(env.render(), dtype=np.uint8).copy()
            goal = np.asarray(info["goal"], dtype=np.uint8).copy()
            raw = {
                "pixels": np.repeat(current[None, None], history_len, axis=1),
                "goal": goal[None, None],
                "action": np.zeros((1, history_len, 2), dtype=np.float32),
            }
            prepared = preprocessor._prepare_info(raw)
            normalized = _vision_door_candidate_bank(
                prepared["pixels"], prepared["goal"],
                horizon=horizon, action_block=action_block,
                action_mean=action_mean, action_std=action_std,
            )[0].detach().cpu()
            model_cost, source_latents, predicted, goal_latents = _model_rollout(
                model, preprocessor, current, goal,
                ActionNormalizer(action_mean, action_std), normalized,
                device, history_len, return_goal=True,
            )
            raw_frames, clipped_frames = _inverse_blocks(
                normalized.numpy(), action_mean, action_std,
                action_low=np.asarray(env.action_space.low, dtype=np.float32),
                action_high=np.asarray(env.action_space.high, dtype=np.float32),
            )
            split = _split_for_root(root_index, int(roots))
            instance_id = f"tworoom_episode_{root_seed}"
            order = np.argsort(model_cost, kind="stable")
            ranks = np.empty(len(order), dtype=np.int64)
            ranks[order] = np.arange(len(order))
            for index, sequence in enumerate(clipped_frames):
                cost, success, terminal_state = _rollout_raw_sequence(env, root_seed, sequence)
                key = f"{instance_id}/{index:03d}"
                predicted_latents[key] = np.asarray(predicted[index], dtype=np.float32)
                if terminal_state is not None:
                    realized_states[key] = terminal_state
                rows.append({
                    "instance_id": instance_id,
                    "split": split,
                    "environment": "TwoRoom",
                    "start_episode_id": str(root_seed),
                    "goal_id": f"tworoom_goal_{root_seed}",
                    "candidate_id": f"candidate_{index:03d}",
                    "candidate_bank_hash": bank_hash,
                    "action_sequence": sequence.tolist(),
                    "normalized_action_sequence": normalized[index].numpy().tolist(),
                    "total_action_steps": int(sequence.shape[0]),
                    "candidate_generator": "vision_door16_runtime_exact",
                    "cem_stage": "rgb_door_bank",
                    "source_observation_ref": f"live_rgb://TwoRoom/{root_seed}/current",
                    "goal_observation_ref": f"live_rgb://TwoRoom/{root_seed}/goal",
                    "source_state_raw": np.asarray(info["state"], dtype=np.float64).tolist(),
                    "goal_raw": np.asarray(info["goal_state"], dtype=np.float64).tolist(),
                    "teacher_model_id": "frozen_lewm_cost_v1",
                    "teacher_score": float(-model_cost[index]),
                    "teacher_rank": int(ranks[index]),
                    "teacher_set_id": f"{instance_id}_vision_door16",
                    "real_success": bool(success),
                    "real_task_cost": float(cost),
                    "realized_observation_ref": f"offline_mujoco_state://{key}",
                    "predicted_latent_ref": f"arrays.npz::predicted::{key}",
                    "realized_latent_ref": None,
                    "realized_terminal_state_ref": f"arrays.npz::realized_state::{key}",
                    "action_support": {
                        "chart": "raw_control",
                        "proposal_distribution": "rgb_vision_door16",
                        "duration_steps": int(sequence.shape[0]),
                    },
                    "privileged_state_role": "offline_outcome_label_only",
                    "source_latent_model": np.asarray(source_latents[index], dtype=np.float64).tolist(),
                    "goal_latent_model": np.asarray(goal_latents[index], dtype=np.float64).tolist(),
                    "predicted_horizon": int(horizon),
                })
            print(f"[vision-door-panel] root={root_index + 1}/{roots} seed={root_seed} candidates={len(clipped_frames)}", flush=True)
    finally:
        env.close()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "arrays.npz",
        **{f"predicted::{key}": value for key, value in predicted_latents.items()},
        **{f"realized_state::{key}": value for key, value in realized_states.items()},
    )
    metadata = {
        "schema": "gdd_wm_tworoom_vision_door_panel.v0.1",
        "environment": "TwoRoom",
        "candidate_bank_hash": bank_hash,
        "candidate_count": 17,
        "history_len": int(history_len),
        "horizon": int(horizon),
        "action_block": int(action_block),
        "root_split": "first_half_train, second_half_evenly_split_calibration_outcome",
        "runtime_rgb_bank_match": True,
        "runtime_state_access": False,
        "outcome_used_for_teacher_or_metric_selection": False,
        "device": device,
        "claim_boundary": "offline RGB door-bank support panel; not closed-loop evidence",
    }
    (output / "candidate_dataset.json").write_text(
        json.dumps({"metadata": metadata, "records": rows}, indent=2, sort_keys=True) + "\n"
    )
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return {"records": len(rows), "roots": roots, "candidate_count": 17, "output": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="/home/gbk/.stable_worldmodel/checkpoints/tworoom/lewm")
    parser.add_argument("--adapter-artifact", type=Path, default=Path("results/live_cross_environment_prediction_20261015/tworoom_cuda/cross_environment_prediction.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20262300)
    parser.add_argument("--roots", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--history-len", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(build(
        policy=args.policy, adapter_artifact=args.adapter_artifact,
        output=args.output, device=args.device, seed=args.seed,
        roots=args.roots, horizon=args.horizon,
        action_block=args.action_block, history_len=args.history_len,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
