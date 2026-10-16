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
from functools import partial
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
from experiments.run_live_tworoom_planning import CandidateBankCEMSolver


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


def _candidate_sequence_bank(
    mean: np.ndarray,
    std: np.ndarray,
    *,
    horizon: int,
    action_block: int,
    bank_family: str,
) -> tuple[np.ndarray, torch.Tensor, list[dict]]:
    """Build the deployment bank while preserving low-level action support."""
    if bank_family == "capacity17":
        raw, normalized = _candidate_bank(
            mean, std, horizon=horizon, action_block=action_block,
        )
        sequences = np.repeat(raw[:, None, :], horizon, axis=1)
        labels = [
            {"family": "constant", "direction": row.tolist(), "phase": None}
            for row in raw
        ]
        return sequences, normalized, labels
    if bank_family not in {"compact", "full"}:
        raise ValueError(f"unknown Reacher candidate bank family: {bank_family}")
    directions = np.asarray(_directions(), dtype=np.float32)
    nonzero = directions[np.linalg.norm(directions, axis=1) > 1.0e-8]
    sequences: list[np.ndarray] = []
    labels: list[dict] = []
    for direction in directions:
        sequences.append(np.repeat(direction[None, :], horizon, axis=0))
        labels.append({"family": "constant", "direction": direction.tolist(), "phase": None})
    if bank_family == "compact":
        pulse_phases = [horizon // 2]
        recovery_phases = [max(0, horizon // 2 - 1)]
    else:
        pulse_phases = list(range(horizon))
        recovery_phases = list(range(horizon - 1))
    for direction in nonzero:
        for phase in pulse_phases:
            sequence = np.zeros((horizon, 2), dtype=np.float32)
            sequence[phase] = direction
            sequences.append(sequence)
            labels.append({"family": "pulse", "direction": direction.tolist(), "phase": phase})
        for phase in recovery_phases:
            sequence = np.zeros((horizon, 2), dtype=np.float32)
            sequence[phase] = direction
            sequence[phase + 1] = -direction
            sequences.append(sequence)
            labels.append({
                "family": "pulse_recovery",
                "direction": direction.tolist(),
                "phase": phase,
            })
    raw_sequences = np.stack(sequences, axis=0)
    normalized_rows = []
    for sequence in raw_sequences:
        blocks = []
        for action in sequence:
            block = np.tile(action[None, :], (action_block, 1)).reshape(-1)
            blocks.append((block - mean) / np.maximum(std, 1.0e-6))
        normalized_rows.append(np.stack(blocks, axis=0))
    return (
        raw_sequences,
        torch.as_tensor(np.stack(normalized_rows), dtype=torch.float32),
        labels,
    )


def _expand_macro_sequence(
    sequence: np.ndarray, *, action_block: int, replay_steps: int,
) -> np.ndarray:
    expanded = np.repeat(np.asarray(sequence, dtype=np.float32), action_block, axis=0)
    if len(expanded) < replay_steps:
        expanded = np.concatenate([
            expanded,
            np.repeat(expanded[-1:], replay_steps - len(expanded), axis=0),
        ], axis=0)
    return expanded[:replay_steps]


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


def _snapshot_reacher_state(env: gym.Env) -> dict[str, np.ndarray]:
    """Capture evaluator-only Reacher state for counterfactual replay."""
    raw = env.unwrapped
    return {
        "physics": np.asarray(raw.env.physics.get_state(), dtype=np.float64).copy(),
        "target_qpos": _target_qpos(env),
    }


def _restore_reacher_state(env: gym.Env, snapshot: dict[str, np.ndarray]) -> None:
    raw = env.unwrapped
    raw.env.physics.set_state(np.asarray(snapshot["physics"], dtype=np.float64))
    raw.env.task.target_qpos = np.asarray(
        snapshot["target_qpos"], dtype=np.float64,
    ).copy()
    raw.env.physics.forward()


def _rollout_candidate_from_snapshot(
    env: gym.Env,
    seed: int,
    snapshot: dict[str, np.ndarray],
    action_steps: np.ndarray,
) -> float:
    """Replay one environment-step action sequence from an online state."""
    env.reset(seed=int(seed))
    _restore_reacher_state(env, snapshot)
    target = np.asarray(snapshot["target_qpos"], dtype=np.float64)
    sequence = np.asarray(action_steps, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[1] != 2:
        raise ValueError(f"candidate action steps must have shape [T,2], got {sequence.shape}")
    for action in sequence:
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    error = _wrap_angle(_current_qpos(env)[:2] - target[:2])
    return float(np.linalg.norm(error))


def _normalized_proposals_to_raw_steps(
    proposals: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    *,
    action_block: int,
) -> np.ndarray:
    """Invert checkpoint normalization without destroying within-block actions."""
    proposals = np.asarray(proposals, dtype=np.float32)
    mean = np.asarray(action_mean, dtype=np.float32)
    std = np.asarray(action_std, dtype=np.float32)
    if proposals.ndim != 3 or proposals.shape[-1] != mean.size or std.shape != mean.shape:
        raise ValueError("proposal and action-statistic shapes are incompatible")
    if mean.size % int(action_block):
        raise ValueError("action statistic width must be divisible by action_block")
    raw_dim = mean.size // int(action_block)
    raw_blocks = proposals * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
    return raw_blocks.reshape(
        proposals.shape[0], proposals.shape[1], int(action_block), raw_dim,
    ).reshape(proposals.shape[0], -1, raw_dim)


def _select_deployment_proposals(
    trace: dict[str, torch.Tensor], count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select outcome-free cost quantiles from the actual Gaussian CEM support."""
    candidates = np.asarray(trace["candidates"], dtype=np.float32)
    costs = np.asarray(trace["costs"], dtype=np.float64)
    bank_mask = np.asarray(trace["bank_mask"], dtype=bool)
    if candidates.ndim != 4 or candidates.shape[0] != 1:
        raise ValueError("deployment proposal collection requires a one-environment trace")
    available = np.flatnonzero(~bank_mask[0])
    if count < 2 or count > len(available):
        raise ValueError(f"proposal count must lie in [2, {len(available)}]")
    ordered = available[np.argsort(costs[0, available], kind="stable")]
    rank_positions = np.rint(np.linspace(0, len(ordered) - 1, count)).astype(np.int64)
    selected = ordered[rank_positions]
    if len(np.unique(selected)) != count:
        raise RuntimeError("cost-quantile proposal selection produced duplicate indices")
    return candidates[0, selected], costs[0, selected], selected


def _deployment_trace_latents(
    trace: dict[str, torch.Tensor], selected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the exact source/predicted/goal tensors used by online scoring."""
    required = {"emb", "predicted_emb", "goal_emb"}
    missing = sorted(required.difference(trace))
    if missing:
        raise RuntimeError(f"online CEM trace is missing latent tensors: {missing}")
    source = np.asarray(trace["emb"])[0, selected, -1]
    predicted = np.asarray(trace["predicted_emb"])[0, selected, -1]
    goal_tensor = np.asarray(trace["goal_emb"])
    if goal_tensor.ndim == 4:
        goal = goal_tensor[0, selected, -1]
    elif goal_tensor.ndim == 3:
        row = goal_tensor[0, -1]
        goal = np.repeat(row[None, :], len(selected), axis=0)
    else:
        raise RuntimeError(f"unexpected traced goal embedding shape: {goal_tensor.shape}")
    return source, predicted, goal


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
        "candidate_bank_count": int(
            args.proposals_per_root
            if args.root_distribution == "euclidean_replan"
            else len(raw_bank)
        ),
        "history_len": int(args.history_len),
        "replay_steps": int(args.replay_steps),
        "target_delta": [float(value) for value in args.target_delta],
        "root_split": "first_half_train, second_half_evenly_split_calibration_outcome",
        "source_panel": (
            "frozen RGB LeWM candidate ranking on actual Euclidean-CEM replan states/proposals"
            if args.root_distribution == "euclidean_replan"
            else "frozen RGB LeWM candidate ranking with simulator labels only"
        ),
        "root_distribution": args.root_distribution,
        "deployment_warmup_steps": int(args.deployment_warmup_steps),
        "proposal_selection": (
            "even frozen-cost quantiles from non-bank candidates in the final CEM iteration"
            if args.root_distribution == "euclidean_replan" else None
        ),
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


def _append_root_result(
    *,
    args: argparse.Namespace,
    root_index: int,
    seed: int,
    instance_id: str,
    source_state_raw: np.ndarray,
    goal_raw: np.ndarray,
    action_steps: np.ndarray,
    candidate_labels: list[dict],
    candidate_generator: str,
    cem_stage: str,
    model_cost: np.ndarray,
    realized_cost: np.ndarray,
    source_latents: np.ndarray,
    predicted_terminal_latents: np.ndarray,
    goal_latents: np.ndarray,
    rows: list[dict],
    panel_rows: list[dict],
    predicted_latents: dict[str, np.ndarray],
) -> None:
    predicted_best = int(np.argmin(model_cost))
    realized_order = np.argsort(realized_cost, kind="stable")
    rows.append({
        "seed": seed,
        "instance_id": instance_id,
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
    })
    if args.panel_output is None:
        return
    split = _split_for_root(root_index, args.roots)
    for candidate_index, sequence in enumerate(action_steps):
        record_key = f"{instance_id}/{candidate_index:03d}"
        predicted_latents[record_key] = np.asarray(
            predicted_terminal_latents[candidate_index], dtype=np.float32,
        )
        panel_rows.append({
            "instance_id": instance_id,
            "split": split,
            "environment": "Reacher",
            "start_episode_id": str(seed),
            "candidate_id": f"candidate_{candidate_index:03d}",
            "candidate_bank_hash": "pending_written_by_panel_writer",
            "action_sequence": np.asarray(sequence, dtype=np.float32).tolist(),
            "total_action_steps": int(len(sequence)),
            "candidate_generator": candidate_generator,
            "candidate_label": candidate_labels[candidate_index],
            "cem_stage": cem_stage,
            "source_observation_ref": f"live_rgb://Reacher/{instance_id}/current",
            "goal_observation_ref": f"live_rgb://Reacher/{instance_id}/goal",
            "source_state_raw": source_state_raw.tolist(),
            "goal_raw": goal_raw.tolist(),
            "teacher_model_id": "frozen_lewm_cost_v1",
            "teacher_score": float(-model_cost[candidate_index]),
            "teacher_rank": int(
                np.flatnonzero(np.argsort(model_cost) == candidate_index)[0]
            ),
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
                "duration_steps": int(len(sequence)),
            },
            "privileged_state_role": "offline_outcome_label_only",
            "source_latent_model": np.asarray(
                source_latents[candidate_index], dtype=np.float64,
            ).tolist(),
            "goal_latent_model": np.asarray(
                goal_latents[candidate_index], dtype=np.float64,
            ).tolist(),
            "predicted_horizon": int(args.horizon),
        })


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
        "--root-distribution", choices=("reset", "euclidean_replan"), default="reset",
        help="reset roots or states reached by the frozen Euclidean CEM controller",
    )
    parser.add_argument(
        "--deployment-warmup-steps", type=int, default=25,
        help="environment steps before collecting a deployment replan state",
    )
    parser.add_argument(
        "--proposals-per-root", type=int, default=64,
        help="outcome-free frozen-cost quantiles retained from final CEM proposals",
    )
    parser.add_argument("--cem-num-samples", type=int, default=256)
    parser.add_argument("--cem-steps", type=int, default=8)
    parser.add_argument("--cem-topk", type=int, default=16)
    parser.add_argument(
        "--bank-family", choices=("capacity17", "compact", "full"),
        default="capacity17",
    )
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
    if args.root_distribution == "euclidean_replan":
        if args.deployment_warmup_steps < 1:
            raise ValueError("deployment warmup must be positive")
        if args.deployment_warmup_steps % (args.horizon * args.action_block):
            raise ValueError(
                "deployment warmup must end at a CEM replan boundary "
                "(a multiple of horizon * action_block)"
            )
        if args.cem_topk >= args.cem_num_samples:
            raise ValueError("CEM topk must be smaller than num samples")

    action_mean, action_std = _load_action_stats(
        args.adapter_artifact, "swm/ReacherDMControl-v0"
    )
    normalizer = ActionNormalizer(action_mean, action_std)
    raw_bank, normalized_bank, candidate_labels = _candidate_sequence_bank(
        action_mean,
        action_std,
        horizon=args.horizon,
        action_block=args.action_block,
        bank_family=args.bank_family,
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
    panel_action_steps: list[np.ndarray] = []
    deployment_world = None
    try:
        if args.root_distribution == "reset":
            fixed_action_steps = np.stack([
                _expand_macro_sequence(
                    action, action_block=args.action_block, replay_steps=args.replay_steps,
                )
                for action in raw_bank
            ])
            for root_index in range(args.roots):
                seed = int(args.seed + root_index)
                _, info = env.reset(seed=seed)
                current = np.asarray(env.render(), dtype=np.uint8).copy()
                goal = np.asarray(info["goal"], dtype=np.uint8).copy()
                source_state_raw = _current_qpos(env)
                goal_raw = _target_qpos(env)
                model_cost, source_latents, predicted_terminal_latents, goal_latents = _model_rollout(
                    model, preprocessor, current, goal, normalized_bank,
                    args.device, args.history_len, return_goal=True,
                )
                realized_cost = np.asarray([
                    _rollout_candidate(
                        env, seed, action, args.replay_steps, args.action_block,
                    )
                    for action in raw_bank
                ], dtype=np.float64)
                _append_root_result(
                    args=args, root_index=root_index, seed=seed,
                    instance_id=f"reacher_episode_{seed}",
                    source_state_raw=source_state_raw, goal_raw=goal_raw,
                    action_steps=fixed_action_steps, candidate_labels=candidate_labels,
                    candidate_generator=f"reacher_{args.bank_family}_sequence_v2",
                    cem_stage="fixed_counterfactual_bank", model_cost=model_cost,
                    realized_cost=realized_cost, source_latents=source_latents,
                    predicted_terminal_latents=predicted_terminal_latents,
                    goal_latents=goal_latents, rows=rows, panel_rows=panel_rows,
                    predicted_latents=predicted_latents,
                )
            panel_action_steps.append(fixed_action_steps)
        else:
            solver = CandidateBankCEMSolver(
                model=model,
                batch_size=1,
                num_samples=args.cem_num_samples,
                var_scale=1.0,
                n_steps=args.cem_steps,
                topk=args.cem_topk,
                device=args.device,
                seed=args.seed,
                candidate_bank=normalized_bank,
                record_candidate_trace=True,
            )
            deployment_policy = swm.policy.WorldModelPolicy(
                solver=solver,
                config=swm.PlanConfig(
                    horizon=args.horizon,
                    receding_horizon=args.horizon,
                    history_len=3,
                    action_block=args.action_block,
                    warm_start=True,
                ),
                process={"action": normalizer},
                transform={"pixels": _image_transform(), "goal": _image_transform()},
            )
            deployment_world = swm.World(
                env_name="swm/ReacherDMControl-v0",
                num_envs=1,
                image_shape=(224, 224),
                max_episode_steps=args.deployment_warmup_steps + 1,
                goal_conditioned=True,
                pre_wrappers=[partial(ReacherGoalImageWrapper, target_delta=args.target_delta)],
                render_mode="rgb_array",
                task="qpos_match",
            )
            deployment_world.set_policy(deployment_policy)
            root_index = 0
            attempt = 0
            max_attempts = args.roots * 4
            while root_index < args.roots and attempt < max_attempts:
                seed = int(args.seed + attempt)
                attempt += 1
                deployment_world.reset(seed=seed)
                for buffer in deployment_policy._action_buffer:
                    buffer.clear()
                deployment_policy._next_init = None
                survived = True
                for _ in range(args.deployment_warmup_steps):
                    action = deployment_world._get_actions()
                    _, rewards, terminateds, truncateds, infos = deployment_world.envs.step(action)
                    deployment_world.rewards = rewards
                    deployment_world.terminateds = terminateds
                    deployment_world.truncateds = truncateds
                    deployment_world.infos = infos
                    if bool(terminateds[0] or truncateds[0]):
                        survived = False
                        break
                if not survived:
                    continue
                if any(len(buffer) for buffer in deployment_policy._action_buffer):
                    raise RuntimeError("warmup did not end on a deployment replan boundary")
                online_env = deployment_world.envs.envs[0]
                snapshot = _snapshot_reacher_state(online_env)
                current = np.asarray(deployment_world.infos["pixels"][0, -1], dtype=np.uint8).copy()
                goal = np.asarray(deployment_world.infos["goal"][0, -1], dtype=np.uint8).copy()
                source_state_raw = _current_qpos(online_env)
                goal_raw = _target_qpos(online_env)
                trace_count = len(solver.candidate_trace)
                deployment_world._get_actions()
                if len(solver.candidate_trace) != trace_count + 1:
                    raise RuntimeError("deployment replan did not emit one candidate trace")
                normalized_proposals, model_cost, selected_indices = _select_deployment_proposals(
                    solver.candidate_trace[-1], args.proposals_per_root,
                )
                raw_action_steps = _normalized_proposals_to_raw_steps(
                    normalized_proposals, action_mean, action_std,
                    action_block=args.action_block,
                )
                source_latents, predicted_terminal_latents, goal_latents = (
                    _deployment_trace_latents(
                        solver.candidate_trace[-1], selected_indices,
                    )
                )
                realized_cost = np.asarray([
                    _rollout_candidate_from_snapshot(env, seed, snapshot, sequence)
                    for sequence in raw_action_steps
                ], dtype=np.float64)
                labels = [
                    {"family": "final_cem_gaussian", "cost_quantile_index": index}
                    for index in range(len(raw_action_steps))
                ]
                _append_root_result(
                    args=args, root_index=root_index, seed=seed,
                    instance_id=f"reacher_episode_{seed}_replan_{args.deployment_warmup_steps}",
                    source_state_raw=source_state_raw, goal_raw=goal_raw,
                    action_steps=raw_action_steps, candidate_labels=labels,
                    candidate_generator="reacher_final_cem_cost_quantiles_v1",
                    cem_stage="final_gaussian_proposal_iteration",
                    model_cost=model_cost, realized_cost=realized_cost,
                    source_latents=source_latents,
                    predicted_terminal_latents=predicted_terminal_latents,
                    goal_latents=goal_latents, rows=rows, panel_rows=panel_rows,
                    predicted_latents=predicted_latents,
                )
                panel_action_steps.append(raw_action_steps)
                root_index += 1
            if root_index != args.roots:
                raise RuntimeError(
                    f"collected only {root_index}/{args.roots} surviving replan roots "
                    f"after {attempt} attempts"
                )
    finally:
        env.close()
        if deployment_world is not None:
            deployment_world.close()

    if args.panel_output is not None:
        _write_candidate_panel(
            args.panel_output,
            args=args,
            raw_bank=np.concatenate(panel_action_steps, axis=0),
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
            "candidate_count": int(
                args.proposals_per_root
                if args.root_distribution == "euclidean_replan"
                else len(raw_bank)
            ),
            "candidate_bank": args.bank_family,
            "root_distribution": args.root_distribution,
            "deployment_warmup_steps": int(args.deployment_warmup_steps),
            "proposal_selection": (
                "even frozen-cost quantiles from non-bank final CEM proposals"
                if args.root_distribution == "euclidean_replan" else None
            ),
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
