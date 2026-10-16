"""Calibration-only latent/action utility audit for Reacher.

The frozen LeWM encoder and predictor produce candidate costs and latent
features.  A low-capacity ridge readout is fitted on calibration roots only
to predict evaluator-only terminal qpos error, then frozen on root-disjoint
held-out roots.  This is an action-identifiability diagnostic; it is not yet
an online planning claim.
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

import numpy as np
import torch

from experiments.audit_live_reacher_shared_candidate_ranking import (
    _candidate_bank,
    _rollout_candidate,
    _spearman,
    _target_qpos,
    _current_qpos,
    _wrap_angle,
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

import gymnasium as gym
import stable_worldmodel as swm
from stable_worldmodel.policy import BasePolicy


@torch.inference_mode()
def _model_latent_features(
    model: torch.nn.Module,
    preprocessor: _Preprocessor,
    current: np.ndarray,
    goal: np.ndarray,
    candidates: torch.Tensor,
    device: str,
    history_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    costs = model.get_cost(model_info, candidate_tensor).squeeze(0)
    predicted = model_info["predicted_emb"][0, :, -1, :]
    goal_embedding = model_info["goal_emb"][0, -1, :].unsqueeze(0).expand_as(predicted)
    current_embedding = model_info["emb"][0, :, -1, :]
    return tuple(
        value.detach().cpu().numpy()
        for value in (costs, predicted, goal_embedding, current_embedding)
    )


def _features(
    costs: np.ndarray,
    predicted: np.ndarray,
    goal: np.ndarray,
    current: np.ndarray,
    raw_actions: np.ndarray,
    feature_family: str = "legacy",
) -> np.ndarray:
    if feature_family not in {"legacy", "effect_geometry"}:
        raise ValueError(f"unknown feature family: {feature_family}")
    z_cost = (costs - costs.mean()) / max(float(costs.std()), 1e-8)
    delta = predicted - goal
    context = current - goal
    base = [z_cost, raw_actions.reshape(len(raw_actions), -1), delta, delta * delta, delta * context]
    if feature_family == "legacy":
        return np.column_stack(base)

    # Low-dimensional action-effect geometry avoids adding another full latent
    # readout.  ``effect`` is the candidate-induced latent displacement and
    # ``desired`` is the current-to-goal displacement; their alignment is the
    # quantity missing from the original candidate feature family.
    effect = predicted - current
    desired = goal - current
    effect_norm = np.linalg.norm(effect, axis=1)
    desired_norm = np.linalg.norm(desired, axis=1)
    residual_norm = np.linalg.norm(effect - desired, axis=1)
    dot = np.sum(effect * desired, axis=1)
    cosine = dot / np.maximum(effect_norm * desired_norm, 1e-8)
    return np.column_stack(base + [
        effect_norm,
        residual_norm,
        dot,
        cosine,
    ])


def _fit_ridge(features: np.ndarray, target: np.ndarray, ridge: float):
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-6)
    normalized = (features - mean) / scale
    design = np.column_stack([normalized, np.ones(len(normalized))])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[-1, -1] = 0.0
    coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return coefficient, mean, scale


def _predict(features: np.ndarray, fit) -> np.ndarray:
    coefficient, mean, scale = fit
    design = np.column_stack([(features - mean) / scale, np.ones(len(features))])
    return design @ coefficient


def _row_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / max(float(values.std()), 1e-8)


def _blend_scores(raw: np.ndarray, calibrated: np.ndarray, alpha: float) -> np.ndarray:
    """Blend calibrated scores while retaining an exact raw-cost fallback."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("blend alpha must be in [0, 1]")
    return (1.0 - float(alpha)) * _row_zscore(raw) + float(alpha) * _row_zscore(calibrated)


def _evaluate(rows: list[dict], indices: np.ndarray, predictions: dict[int, np.ndarray]) -> dict:
    top1, top4, ranks, regrets = [], [], [], []
    for index in indices.tolist():
        row = rows[index]
        predicted = predictions[index]
        realized = np.asarray(row["realized_cost"], dtype=np.float64)
        best = int(np.argmin(predicted))
        order = np.argsort(realized, kind="stable")
        top1.append(int(best == int(order[0])))
        top4.append(int(best in set(order[:4].tolist())))
        ranks.append(int(np.flatnonzero(order == best)[0] + 1))
        regrets.append(float(realized[best] - realized[order[0]]))
    return {
        "n": int(len(indices)),
        "top1_accuracy": float(np.mean(top1)),
        "top4_accuracy": float(np.mean(top4)),
        "mean_true_rank_of_predicted_best": float(np.mean(ranks)),
        "mean_regret": float(np.mean(regrets)),
        "median_regret": float(np.median(regrets)),
        "top1_hits": int(np.sum(top1)),
    }


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
    parser.add_argument("--seed", type=int, default=20261420)
    parser.add_argument("--roots", type=int, default=32)
    parser.add_argument("--calibration-roots", type=int, default=16)
    parser.add_argument("--replay-steps", type=int, default=25)
    parser.add_argument("--history-len", type=int, default=3)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=100.0)
    parser.add_argument(
        "--feature-family",
        choices=("legacy", "effect_geometry"),
        default="legacy",
        help="candidate feature family; effect_geometry adds frozen action-effect alignment summaries",
    )
    parser.add_argument("--target-delta", type=float, nargs=2, default=(0.65, -0.65))
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not 1 <= args.calibration_roots < args.roots:
        raise ValueError("calibration roots must be positive and smaller than total roots")

    mean, std = _load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    normalizer = ActionNormalizer(mean, std)
    raw_bank, normalized_bank = _candidate_bank(
        mean, std, horizon=args.horizon, action_block=args.action_block
    )
    raw_actions = np.asarray(raw_bank, dtype=np.float64)
    model = load_compatible_pretrained(args.policy, device=args.device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    preprocessor = _Preprocessor(normalizer)
    env = ReacherGoalImageWrapper(
        gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"),
        target_delta=args.target_delta,
    )
    rows: list[dict] = []
    feature_rows: list[np.ndarray] = []
    try:
        for root_index in range(args.roots):
            seed = int(args.seed + root_index)
            _, info = env.reset(seed=seed)
            current = np.asarray(env.render(), dtype=np.uint8).copy()
            goal = np.asarray(info["goal"], dtype=np.uint8).copy()
            costs, predicted, goal_emb, current_emb = _model_latent_features(
                model, preprocessor, current, goal, normalized_bank, args.device, args.history_len
            )
            realized = np.asarray(
                [
                    _rollout_candidate(
                        env, seed, np.tile(action[None, :], (args.action_block, 1)),
                        args.replay_steps, args.action_block,
                    )
                    for action in raw_bank
                ],
                dtype=np.float64,
            )
            feature_rows.append(_features(
                costs, predicted, goal_emb, current_emb, raw_actions,
                feature_family=args.feature_family,
            ))
            rows.append({
                "seed": seed,
                "model_cost": costs.tolist(),
                "realized_cost": realized.tolist(),
                "raw_spearman": _spearman(costs, realized),
            })
    finally:
        env.close()

    calibration = np.arange(args.calibration_roots, dtype=np.int64)
    held_out = np.arange(args.calibration_roots, len(rows), dtype=np.int64)
    fit = _fit_ridge(
        np.concatenate([feature_rows[index] for index in calibration], axis=0),
        np.concatenate([np.asarray(rows[index]["realized_cost"]) for index in calibration]),
        args.ridge,
    )
    ranker_scores = {index: _predict(feature_rows[index], fit) for index in range(len(rows))}
    raw_scores = {index: np.asarray(rows[index]["model_cost"], dtype=np.float64) for index in range(len(rows))}
    summary = {
        "calibration_raw": _evaluate(rows, calibration, raw_scores),
        "calibration_ranker": _evaluate(rows, calibration, ranker_scores),
        "held_out_raw": _evaluate(rows, held_out, raw_scores),
        "held_out_ranker": _evaluate(rows, held_out, ranker_scores),
    }
    summary["held_out_delta_ranker_minus_raw_regret"] = (
        summary["held_out_ranker"]["mean_regret"] - summary["held_out_raw"]["mean_regret"]
    )
    summary["held_out_delta_ranker_minus_raw_top1"] = (
        summary["held_out_ranker"]["top1_accuracy"] - summary["held_out_raw"]["top1_accuracy"]
    )

    # Nested calibration for a conservative intervention.  The first half of
    # calibration roots fits the readout; the second half selects alpha from a
    # fixed grid.  The final held-out roots are untouched by both choices.
    split = max(1, len(calibration) // 2)
    fit_roots = calibration[:split]
    select_roots = calibration[split:]
    if len(select_roots) == 0:
        raise ValueError("nested calibration requires at least two calibration roots")
    nested_fit = _fit_ridge(
        np.concatenate([feature_rows[index] for index in fit_roots], axis=0),
        np.concatenate([np.asarray(rows[index]["realized_cost"]) for index in fit_roots]),
        args.ridge,
    )
    nested_predictions = {
        index: _predict(feature_rows[index], nested_fit) for index in range(len(rows))
    }
    alpha_grid = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
    selection_scores = {}
    for alpha in alpha_grid:
        selection_scores[alpha] = _evaluate(
            rows,
            select_roots,
            {
                index: _blend_scores(
                    np.asarray(rows[index]["model_cost"], dtype=np.float64),
                    nested_predictions[index],
                    alpha,
                )
                for index in select_roots.tolist()
            },
        )
    selected_alpha = min(
        alpha_grid,
        key=lambda alpha: (selection_scores[alpha]["mean_regret"], alpha),
    )
    nested_scores = {
        index: _blend_scores(
            np.asarray(rows[index]["model_cost"], dtype=np.float64),
            nested_predictions[index],
            selected_alpha,
        )
        for index in range(len(rows))
    }
    nested_held_out = _evaluate(rows, held_out, nested_scores)
    summary["nested_safe_blend"] = {
        "fit": _evaluate(rows, fit_roots, nested_scores),
        "selection": _evaluate(rows, select_roots, nested_scores),
        "held_out": nested_held_out,
        "held_out_delta_blend_minus_raw_regret": (
            nested_held_out["mean_regret"] - summary["held_out_raw"]["mean_regret"]
        ),
        "held_out_delta_blend_minus_raw_top1": (
            nested_held_out["top1_accuracy"] - summary["held_out_raw"]["top1_accuracy"]
        ),
        "selected_alpha": float(selected_alpha),
        "alpha_grid": [float(alpha) for alpha in alpha_grid],
        "selection_scores": {str(alpha): value for alpha, value in selection_scores.items()},
        "fail_closed_alpha_zero": True,
        "nested_fit_roots": [int(rows[index]["seed"]) for index in fit_roots],
        "nested_selection_roots": [int(rows[index]["seed"]) for index in select_roots],
    }
    output = {
        "schema": "reacher_latent_action_ranker.v2",
        "protocol": {
            "checkpoint_frozen": True,
            "candidate_bank_shared": True,
            "root_disjoint": True,
            "calibration_roots": [int(rows[index]["seed"]) for index in calibration],
            "held_out_roots": [int(rows[index]["seed"]) for index in held_out],
            "history_len": int(args.history_len),
            "target_delta": [float(value) for value in args.target_delta],
            "runtime_state_access": False,
            "simulator_labels": "calibration/evaluation only",
            "test_planning_used_for_selection": False,
            "device": args.device,
            "ridge": float(args.ridge),
            "feature_family": args.feature_family,
            "nested_safe_blend": "fit on first calibration half, select alpha on second calibration half, evaluate on held-out roots",
        },
        "summary": summary,
        "claim_boundary": "Calibration-only frozen-latent utility diagnostic; no online planning or universal transfer claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
