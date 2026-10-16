"""Root-disjoint Reacher ranking audit with a fixed two-phase action bank.

The earlier shared bank contained only constant torques repeated for the full
replay.  This audit keeps the frozen RGB LeWM and evaluator protocol fixed but
adds predeclared pulse and pulse-recovery sequences.  Every sequence is scored
by the model and replayed from the same root, so the result separates action
support from latent ranking quality rather than silently changing the labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in (None, ""):
    import sys

    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

import gymnasium as gym
import numpy as np
import torch
from stable_worldmodel.policy import BasePolicy

from experiments.audit_live_reacher_shared_candidate_ranking import (
    _model_costs,
    _pairwise_inversion,
    _rollout_candidate,
    _spearman,
    _directions,
    _Preprocessor,
)
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from experiments.run_live_reacher_planning import (
    ReacherGoalImageWrapper,
    _image_transform,
    _load_action_stats,
    install_missing_field_compat,
)


def _sequence_bank(
    mean: np.ndarray,
    std: np.ndarray,
    *,
    horizon: int,
    action_block: int,
    bank_family: str = "full",
) -> tuple[np.ndarray, torch.Tensor, list[dict]]:
    """Build constant, pulse, and pulse-recovery candidates deterministically."""
    if horizon < 2 or action_block < 1:
        raise ValueError("two-phase bank requires horizon >= 2 and action_block >= 1")
    if bank_family not in {"full", "compact", "support4"}:
        raise ValueError(f"unknown bank family: {bank_family}")
    directions = np.asarray(_directions(), dtype=np.float32)
    nonzero = directions[np.linalg.norm(directions, axis=1) > 1e-8]
    sequences: list[np.ndarray] = []
    labels: list[dict] = []
    if bank_family != "support4":
        for direction in directions:
            sequences.append(np.repeat(direction[None, :], horizon, axis=0))
            labels.append({"family": "constant", "direction": direction.tolist(), "phase": None})
    if bank_family == "full":
        pulse_phases = list(range(horizon))
        recovery_phases = list(range(horizon - 1))
        pulse_directions = nonzero
    elif bank_family == "compact":
        # Keep the bank small enough that CEM retains most of its continuous
        # exploration budget.  The midpoint phase is fixed before evaluation.
        midpoint = horizon // 2
        pulse_phases = [midpoint]
        recovery_phases = [max(0, midpoint - 1)]
        pulse_directions = nonzero
    else:
        # A four-candidate support probe preserves essentially all Gaussian
        # exploration budget.  The cardinal +/-0.5 directions and the
        # midpoint pulse-recovery phase are frozen before any test outcome.
        midpoint = horizon // 2
        pulse_phases = []
        recovery_phases = [max(0, midpoint - 1)]
        pulse_directions = np.asarray(
            [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5)],
            dtype=np.float32,
        )
    for direction in pulse_directions:
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
            labels.append({"family": "pulse_recovery", "direction": direction.tolist(), "phase": phase})

    normalized_rows = []
    for sequence in sequences:
        rows = []
        for frame in sequence:
            block = np.tile(frame[None, :], (action_block, 1)).reshape(-1)
            rows.append((block - mean) / np.maximum(std, 1e-6))
        normalized_rows.append(np.stack(rows, axis=0))
    return (
        np.stack(sequences, axis=0),
        torch.as_tensor(np.stack(normalized_rows, axis=0), dtype=torch.float32),
        labels,
    )


def _bootstrap(values: np.ndarray, seed: int, draws: int = 4000) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return [float(values.mean()), float(values.mean())]
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def main() -> None:
    install_missing_field_compat()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="/home/gbk/.stable_worldmodel/checkpoints/reacher/lewm")
    parser.add_argument(
        "--adapter-artifact", type=Path,
        default=Path("results/live_cross_environment_prediction_20261015/reacher_cuda/cross_environment_prediction.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261500)
    parser.add_argument("--roots", type=int, default=32)
    parser.add_argument("--replay-steps", type=int, default=25)
    parser.add_argument("--history-len", type=int, default=3)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--bank-family", choices=("full", "compact"), default="full")
    parser.add_argument("--target-delta", type=float, nargs=2, default=(0.65, -0.65))
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.roots < 2:
        raise ValueError("roots must be >= 2")

    mean, std = _load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    normalizer = ActionNormalizer(mean, std)
    raw_sequences, normalized_bank, labels = _sequence_bank(
        mean, std, horizon=args.horizon, action_block=args.action_block,
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
    try:
        for root_index in range(args.roots):
            seed = int(args.seed + root_index)
            _, info = env.reset(seed=seed)
            current = np.asarray(env.render(), dtype=np.uint8).copy()
            goal = np.asarray(info["goal"], dtype=np.uint8).copy()
            model_cost = _model_costs(
                model, preprocessor, current, goal, normalized_bank,
                args.device, args.history_len,
            )
            realized = np.asarray([
                _rollout_candidate(
                    env, seed, sequence, args.replay_steps, args.action_block,
                )
                for sequence in raw_sequences
            ], dtype=np.float64)
            order = np.argsort(realized, kind="stable")
            selected = int(np.argmin(model_cost))
            rows.append({
                "seed": seed,
                "candidate_count": int(len(labels)),
                "top1_hit": int(selected == int(order[0])),
                "top4_hit": int(selected in set(order[:4].tolist())),
                "true_rank": int(np.flatnonzero(order == selected)[0] + 1),
                "planning_regret": float(realized[selected] - realized[order[0]]),
                "multistep_task_error": float(realized[selected]),
                "action_ranking_inversion": _pairwise_inversion(model_cost, realized),
                "spearman": _spearman(model_cost, realized),
                "selected_label": labels[selected],
                "oracle_cost": float(realized[order[0]]),
            })
    finally:
        env.close()

    regret = np.asarray([row["planning_regret"] for row in rows], dtype=np.float64)
    inversion = np.asarray([row["action_ranking_inversion"] for row in rows], dtype=np.float64)
    top1 = np.asarray([row["top1_hit"] for row in rows], dtype=np.float64)
    top4 = np.asarray([row["top4_hit"] for row in rows], dtype=np.float64)
    payload = {
        "schema": "reacher_sequence_candidate_ranking.v1",
        "protocol": {
            "checkpoint_frozen": True,
            "candidate_bank_shared": True,
            "candidate_bank": args.bank_family,
            "candidate_count": int(len(labels)),
            "root_disjoint": True,
            "history_len": int(args.history_len),
            "target_delta": [float(value) for value in args.target_delta],
            "roots": int(args.roots),
            "replay_steps": int(args.replay_steps),
            "runtime_state_access": False,
            "simulator_labels": "evaluation only",
            "test_planning_used_for_selection": False,
            "device": args.device,
        },
        "summary": {
            "n": int(len(rows)),
            "top1_accuracy": float(top1.mean()),
            "top1_hits": int(top1.sum()),
            "top4_accuracy": float(top4.mean()),
            "mean_true_rank": float(np.mean([row["true_rank"] for row in rows])),
            "mean_action_ranking_inversion": float(inversion.mean()),
            "action_ranking_inversion_ci95": _bootstrap(inversion, args.seed + 1),
            "mean_planning_regret": float(regret.mean()),
            "planning_regret_ci95": _bootstrap(regret, args.seed + 2),
            "mean_multistep_task_error": float(np.mean([row["multistep_task_error"] for row in rows])),
            "mean_spearman": float(np.mean([row["spearman"] for row in rows])),
        },
        "rows": rows,
        "claim_boundary": "Root-disjoint Reacher action-support diagnostic; no universal latent dynamics or online planning claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
