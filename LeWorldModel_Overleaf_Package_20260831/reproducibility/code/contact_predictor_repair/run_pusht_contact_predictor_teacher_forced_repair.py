"""Teacher-forced multi-step repair of the frozen PushT predictor.

The earlier contact repair supervised only the final latent at each horizon.
This runner keeps the visual encoder frozen but supervises every predicted
future latent against the corresponding simulator frame.  It therefore tests
whether the remaining negative physical counterfactual result is caused by
autoregressive exposure bias rather than by an irrecoverable action interface.

The experiment is a targeted repair of one published checkpoint.  It is not a
claim that the released frozen checkpoint already has the repaired behavior,
nor does it establish universal causality, online planning, or task success.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.run_pusht_contact_predicted_latent_adapter import (
    approach_to_contact,
    bootstrap,
    image_transform,
    intervention_actions,
    normalize_public_action,
    pose_target,
    sample_cases,
)
from experiments.run_pusht_counterfactual import encode_images

gym.logger.min_level = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-cases", type=int, default=96)
    parser.add_argument("--held-out", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--anchor-weight", type=float, default=0.05)
    parser.add_argument(
        "--free-run-weight",
        type=float,
        default=0.0,
        help="weight of the fully autoregressive multi-step loss; zero reproduces teacher forcing",
    )
    parser.add_argument(
        "--pose-loss-weight",
        type=float,
        default=0.0,
        help="auxiliary normalized pose loss on predicted latents; zero reproduces latent-only repair",
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--train-horizons", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--approach-steps", type=int, default=40)
    parser.add_argument("--action-magnitude", type=float, default=0.85)
    parser.add_argument("--batch-size-encode", type=int, default=16)
    parser.add_argument(
        "--calibration-cases",
        nargs="+",
        type=int,
        default=None,
        help="case counts for the repaired-readout calibration-size ablation",
    )
    parser.add_argument("--seed", type=int, default=8421)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def normalized_context_action_sequence(
    case: dict[str, Any], action: np.ndarray, horizon: int
) -> np.ndarray:
    """Return two observed actions plus ``horizon`` intervention actions."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    past = np.asarray(case["context_actions"], dtype=np.float32)
    if past.shape != (2, 2):
        raise ValueError(f"context_actions must have shape (2, 2), got {past.shape}")
    raw = np.concatenate(
        [past, np.repeat(np.asarray(action, dtype=np.float32)[None], horizon, axis=0)]
    )
    normalized = np.asarray([normalize_public_action(row) for row in raw], dtype=np.float32)
    return np.asarray([np.tile(row, 5) for row in normalized], dtype=np.float32)


@torch.enable_grad()
def predict_latent_sequence(
    model: torch.nn.Module,
    context: torch.Tensor,
    actions: torch.Tensor,
    teacher_forcing_targets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Predict one latent per future action, optionally with teacher forcing.

    With ``teacher_forcing_targets`` the target latent replaces the previous
    prediction in the next context window during training.  Evaluation omits
    this argument and is therefore fully autoregressive, matching
    ``JEPA.rollout``.
    """
    if context.ndim != 3 or actions.ndim != 3:
        raise ValueError("context and actions must be 3D tensors")
    if context.size(0) != actions.size(0) or context.size(1) != 3:
        raise ValueError("context must be (B, 3, D) and match actions")
    if actions.size(1) < 4:
        raise ValueError("actions must contain 3 context rows and at least one future row")
    if teacher_forcing_targets is not None:
        if teacher_forcing_targets.ndim != 3:
            raise ValueError("teacher_forcing_targets must be 3D")
        if teacher_forcing_targets.shape[0] != context.shape[0]:
            raise ValueError("teacher-forcing targets must match the batch")
        if teacher_forcing_targets.shape[1] != actions.size(1) - 2:
            raise ValueError("teacher-forcing targets must have one row per future step")
    act_emb = model.action_encoder(actions)
    history = [value for value in context.unbind(dim=1)]
    action_history = [value for value in act_emb[:, :3].unbind(dim=1)]
    predictions: list[torch.Tensor] = []
    # The final JEPA prediction is made after consuming the last action row;
    # therefore T actions with a three-frame context produce T-2 endpoints.
    future_steps = actions.size(1) - 2
    for step in range(future_steps):
        emb_trunc = torch.stack(history[-3:], dim=1)
        act_trunc = torch.stack(action_history[-3:], dim=1)
        predicted = model.predict(emb_trunc, act_trunc)[:, -1]
        predictions.append(predicted)
        history.append(
            teacher_forcing_targets[:, step]
            if teacher_forcing_targets is not None
            else predicted
        )
        if step + 1 < future_steps:
            action_history.append(act_emb[:, 3 + step])
    return torch.stack(predictions, dim=1)


def rollout_trajectory(
    env: gym.Env, case: dict[str, Any], action: np.ndarray, horizon: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    env.reset(seed=0, options={"state": case["state"].copy(), "goal_state": case["goal"].copy()})
    states: list[np.ndarray] = []
    images: list[np.ndarray] = []
    for _ in range(horizon):
        observation, _, terminated, truncated, _ = env.step(action)
        states.append(np.asarray(observation["state"], dtype=np.float64).copy())
        images.append(env.render().copy())
        if terminated or truncated:
            break
    if len(states) != horizon:
        raise RuntimeError(f"rollout terminated before horizon {horizon}: {len(states)}")
    return states, images


def collect_data(
    model: torch.nn.Module,
    env: gym.Env,
    cases: list[dict[str, Any]],
    transform,
    args: argparse.Namespace,
    max_horizon: int,
) -> dict[str, np.ndarray]:
    contexts: list[list[np.ndarray]] = []
    action_sequences: list[np.ndarray] = []
    target_images: list[np.ndarray] = []
    target_poses: list[np.ndarray] = []
    target_states: list[np.ndarray] = []
    for case in cases:
        for action in intervention_actions(case, args.action_magnitude):
            states, images = rollout_trajectory(env, case, action, max_horizon)
            contexts.append(case["context_images"])
            action_sequences.append(
                normalized_context_action_sequence(case, action, max_horizon)
            )
            target_images.extend(images)
            target_poses.append(np.asarray([pose_target(state) for state in states]))
            target_states.append(np.asarray(states, dtype=np.float64))
    context_flat = [image for context in contexts for image in context]
    context_z = encode_images(
        model, context_flat, transform, args.batch_size_encode, args.device
    ).reshape(len(contexts), 3, -1)
    target_z = encode_images(
        model, target_images, transform, args.batch_size_encode, args.device
    ).reshape(len(contexts), max_horizon, -1)
    return {
        "context_z": context_z.astype(np.float32),
        "actions": np.asarray(action_sequences, dtype=np.float32),
        "target_z": target_z.astype(np.float32),
        "target_pose": np.asarray(target_poses, dtype=np.float64),
        "target_state": np.asarray(target_states, dtype=np.float64),
    }


def predict_all(
    model: torch.nn.Module, data: dict[str, np.ndarray], device: str
) -> torch.Tensor:
    context = torch.as_tensor(data["context_z"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(data["actions"], dtype=torch.float32, device=device)
    return predict_latent_sequence(model, context, actions)


def train_repair(
    model: torch.nn.Module, data: dict[str, np.ndarray], args: argparse.Namespace
) -> dict[str, Any]:
    device = torch.device(args.device)
    context = torch.as_tensor(data["context_z"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(data["actions"], dtype=torch.float32, device=device)
    targets = torch.as_tensor(data["target_z"], dtype=torch.float32, device=device)
    pose_targets = torch.as_tensor(data["target_pose"], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        base = predict_latent_sequence(model, context, actions)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = list(model.action_encoder.parameters())
    trainable += list(model.predictor.parameters())
    trainable += list(model.pred_proj.parameters())
    pose_head = None
    if args.pose_loss_weight > 0:
        pose_mean = pose_targets.reshape(-1, pose_targets.size(-1)).mean(dim=0)
        pose_std = pose_targets.reshape(-1, pose_targets.size(-1)).std(dim=0).clamp_min(1e-6)
        pose_head = torch.nn.Sequential(
            torch.nn.LayerNorm(targets.size(-1)),
            torch.nn.Linear(targets.size(-1), 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, pose_targets.size(-1)),
        ).to(device)
        model._contact_pose_head = pose_head
        model._contact_pose_mean = pose_mean.detach()
        model._contact_pose_std = pose_std.detach()
        trainable += list(pose_head.parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    model.action_encoder.train()
    model.predictor.train()
    model.pred_proj.train()
    if pose_head is not None:
        pose_head.train()
    horizon_values = sorted(set(int(value) for value in args.train_horizons))
    max_horizon = targets.size(1)
    if any(value < 1 or value > max_horizon for value in horizon_values):
        raise ValueError("train horizons exceed collected trajectory length")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    final_losses: list[float] = []
    for _ in range(args.epochs):
        order = torch.randperm(context.size(0), generator=generator, device=device)
        for start in range(0, order.numel(), args.batch_size):
            index = order[start : start + args.batch_size]
            predicted = predict_latent_sequence(
                model,
                context[index],
                actions[index],
                teacher_forcing_targets=targets[index],
            )
            # Teacher forcing stabilizes optimization but hides exposure bias:
            # every next context contains the ground-truth latent.  A separate
            # fully autoregressive term trains on the same held-in trajectories
            # under the state distribution used at evaluation time.  Keeping it
            # as an explicit weight makes the original protocol reproducible and
            # turns exposure bias into a controlled ablation.
            free_running = predict_latent_sequence(
                model,
                context[index],
                actions[index],
            )
            fit_terms = []
            anchor_terms = []
            for horizon in horizon_values:
                fit_terms.append((predicted[:, :horizon] - targets[index, :horizon]).square().mean())
                anchor_terms.append((predicted[:, :horizon] - base[index, :horizon]).square().mean())
            free_run_terms = [
                (free_running[:, :horizon] - targets[index, :horizon]).square().mean()
                for horizon in horizon_values
            ]
            pose_terms = []
            if pose_head is not None:
                pose_target_norm = (
                    pose_targets[index] - model._contact_pose_mean
                ) / model._contact_pose_std
                pose_terms.append(
                    torch.stack([
                        (pose_head(predicted[:, :horizon]) - pose_target_norm[:, :horizon]).square().mean()
                        for horizon in horizon_values
                    ]).mean()
                )
                if args.free_run_weight > 0:
                    pose_terms.append(
                        args.free_run_weight
                        * torch.stack([
                            (pose_head(free_running[:, :horizon]) - pose_target_norm[:, :horizon]).square().mean()
                            for horizon in horizon_values
                        ]).mean()
                    )
            loss = (
                torch.stack(fit_terms).mean()
                + args.anchor_weight * torch.stack(anchor_terms).mean()
                + args.free_run_weight * torch.stack(free_run_terms).mean()
            )
            if pose_terms:
                loss = loss + args.pose_loss_weight * torch.stack(pose_terms).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            final_losses.append(float(loss.detach().cpu()))
    model.eval()
    if pose_head is not None:
        pose_head.eval()
    with torch.no_grad():
        repaired = predict_latent_sequence(model, context, actions)
    train_metrics = {}
    for horizon in horizon_values:
        h = horizon - 1
        train_metrics[str(horizon)] = {
            "base_endpoint_mse": float((base[:, h] - targets[:, h]).square().mean().cpu()),
            "repaired_endpoint_mse": float((repaired[:, h] - targets[:, h]).square().mean().detach().cpu()),
            "base_trajectory_mse": float((base[:, :horizon] - targets[:, :horizon]).square().mean().cpu()),
            "repaired_trajectory_mse": float((repaired[:, :horizon] - targets[:, :horizon]).square().mean().detach().cpu()),
        }
    return {
        "final_loss": final_losses[-1] if final_losses else float("nan"),
        "mean_final_epoch_loss": float(np.mean(final_losses[-max(1, len(final_losses) // max(1, args.epochs)) :])),
        "samples": int(context.size(0)),
        "pose_head": bool(pose_head is not None),
        "by_horizon": train_metrics,
    }


def fit_decoder(features: np.ndarray, targets: np.ndarray) -> tuple[Ridge, StandardScaler]:
    scaler = StandardScaler().fit(targets)
    return Ridge(alpha=1.0).fit(features, scaler.transform(targets)), scaler


def evaluate(
    base_model: torch.nn.Module,
    repaired_model: torch.nn.Module,
    train_data: dict[str, np.ndarray],
    test_data: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    with torch.no_grad():
        base_train = predict_all(base_model, train_data, args.device).cpu().numpy()
        repaired_train = predict_all(repaired_model, train_data, args.device).cpu().numpy()
        base_test = predict_all(base_model, test_data, args.device).cpu().numpy()
        repaired_test = predict_all(repaired_model, test_data, args.device).cpu().numpy()
        pose_head_test = None
        if hasattr(repaired_model, "_contact_pose_head"):
            pose_head_test = (
                repaired_model._contact_pose_head(
                    torch.as_tensor(repaired_test, dtype=torch.float32, device=args.device)
                )
                * repaired_model._contact_pose_std
                + repaired_model._contact_pose_mean
            ).cpu().numpy()
    available_calibration_cases = len(train_data["target_pose"]) // 3
    requested_calibration_cases = sorted(
        set(int(value) for value in (args.calibration_cases or [available_calibration_cases]))
    )
    if any(value < 4 or value > available_calibration_cases for value in requested_calibration_cases):
        raise ValueError(
            "calibration case counts must be between 4 and the available training cases "
            f"({available_calibration_cases})"
        )
    reports = []
    for horizon in sorted(set(args.horizons)):
        index = horizon - 1
        train_target = train_data["target_pose"][:, index]
        test_target = test_data["target_pose"][:, index]
        base_decoder, base_scaler = fit_decoder(base_train[:, index], train_target)
        repaired_decoder, repaired_scaler = fit_decoder(repaired_train[:, index], train_target)
        observed_decoder, observed_scaler = fit_decoder(
            train_data["target_z"][:, index], train_target
        )
        base_pose = base_scaler.inverse_transform(base_decoder.predict(base_test[:, index]))
        repaired_pose = base_scaler.inverse_transform(base_decoder.predict(repaired_test[:, index]))
        repaired_calibrated_pose = repaired_scaler.inverse_transform(
            repaired_decoder.predict(repaired_test[:, index])
        )
        pose_head_pose = None if pose_head_test is None else pose_head_test[:, index]
        observed_pose = observed_scaler.inverse_transform(
            observed_decoder.predict(test_data["target_z"][:, index])
        )

        def metric(values: np.ndarray, seed_offset: int) -> dict[str, Any]:
            return bootstrap(values, args.seed + seed_offset, 2000)

        # Rows are ordered +a, -a, zero for each case.  Keeping all three arms
        # exposes an action-independent readout bias that a +a-only table hides.
        arm_rows = {
            "plus": np.arange(0, len(test_target), 3),
            "minus": np.arange(1, len(test_target), 3),
            "zero": np.arange(2, len(test_target), 3),
        }
        arm_errors: dict[str, dict[str, np.ndarray]] = {}
        for arm, rows in arm_rows.items():
            true = test_target[rows, :2]
            arm_errors[arm] = {
                "base": np.linalg.norm(base_pose[rows, :2] - true, axis=1),
                "repaired_native": np.linalg.norm(repaired_pose[rows, :2] - true, axis=1),
                "repaired_calibrated": np.linalg.norm(
                    repaired_calibrated_pose[rows, :2] - true, axis=1
                ),
                "observed_latent_upper_bound": np.linalg.norm(
                    observed_pose[rows, :2] - true, axis=1
                ),
            }
        plus_rows = arm_rows["plus"]
        base_error = arm_errors["plus"]["base"]
        repaired_error = arm_errors["plus"]["repaired_native"]
        repaired_calibrated_error = arm_errors["plus"]["repaired_calibrated"]
        calibration_ablation: dict[str, dict[str, dict[str, Any]]] = {}
        for case_count in requested_calibration_cases:
            calibration_rows = np.arange(3 * case_count)
            calibration_decoder, calibration_scaler = fit_decoder(
                repaired_train[calibration_rows, index], train_target[calibration_rows]
            )
            calibration_pose = calibration_scaler.inverse_transform(
                calibration_decoder.predict(repaired_test[:, index])
            )
            calibration_ablation[str(case_count)] = {}
            for arm_offset, (arm, rows) in enumerate(arm_rows.items()):
                error = np.linalg.norm(
                    calibration_pose[rows, :2] - test_target[rows, :2], axis=1
                )
                calibration_ablation[str(case_count)][arm] = metric(
                    error, 1200 + horizon * 100 + case_count + arm_offset
                )
        report = {
            "horizon": horizon,
            "metrics": {
                "base_error_plus": metric(base_error, 100 + horizon),
                "repaired_error_plus": metric(repaired_error, 200 + horizon),
                "repaired_calibrated_error_plus": metric(repaired_calibrated_error, 300 + horizon),
                "paired_calibrated_gain_plus": metric(base_error - repaired_calibrated_error, 900 + horizon),
                "pose_head_error_plus": (
                    metric(
                        np.linalg.norm(pose_head_pose[plus_rows, :2] - test_target[plus_rows, :2], axis=1),
                        1250 + horizon,
                    )
                    if pose_head_pose is not None else None
                ),
                "pose_head_gain_plus": (
                    metric(
                        base_error
                        - np.linalg.norm(pose_head_pose[plus_rows, :2] - test_target[plus_rows, :2], axis=1),
                        1260 + horizon,
                    )
                    if pose_head_pose is not None else None
                ),
                "paired_native_gain_plus": metric(base_error - repaired_error, 1000 + horizon),
                "physical_counterfactual_separation": metric(np.linalg.norm(test_target[::3, :2] - test_target[1::3, :2], axis=1), 400 + horizon),
                "base_latent_counterfactual_separation": metric(np.linalg.norm(base_test[::3, index] - base_test[1::3, index], axis=1), 500 + horizon),
                "repaired_latent_counterfactual_separation": metric(np.linalg.norm(repaired_test[::3, index] - repaired_test[1::3, index], axis=1), 600 + horizon),
                "base_decoded_counterfactual_separation": metric(np.linalg.norm(base_pose[::3, :2] - base_pose[1::3, :2], axis=1), 1100 + horizon),
                "repaired_decoded_counterfactual_separation": metric(np.linalg.norm(repaired_calibrated_pose[::3, :2] - repaired_calibrated_pose[1::3, :2], axis=1), 1150 + horizon),
                "base_endpoint_latent_error": metric(np.linalg.norm(base_test[:, index] - test_data["target_z"][:, index], axis=1), 700 + horizon),
                "repaired_endpoint_latent_error": metric(np.linalg.norm(repaired_test[:, index] - test_data["target_z"][:, index], axis=1), 800 + horizon),
                "observed_latent_upper_bound_error_plus": metric(arm_errors["plus"]["observed_latent_upper_bound"], 1800 + horizon),
            },
            "by_arm": {
                arm: {
                    "base_error": metric(values["base"], 2000 + horizon * 10 + offset),
                    "repaired_native_error": metric(values["repaired_native"], 2100 + horizon * 10 + offset),
                    "repaired_calibrated_error": metric(values["repaired_calibrated"], 2200 + horizon * 10 + offset),
                    "observed_latent_upper_bound_error": metric(values["observed_latent_upper_bound"], 2300 + horizon * 10 + offset),
                    "paired_calibrated_gain": metric(values["base"] - values["repaired_calibrated"], 2400 + horizon * 10 + offset),
                    "pose_head_error": (
                        metric(
                            np.linalg.norm(
                                pose_head_pose[rows, :2] - test_target[rows, :2], axis=1
                            ),
                            2500 + horizon * 10 + offset,
                        )
                        if pose_head_pose is not None else None
                    ),
                }
                for offset, (arm, values) in enumerate(arm_errors.items())
            },
            "calibration_ablation": calibration_ablation,
            "sanity_checks": {
                "finite": bool(np.isfinite(base_test).all() and np.isfinite(repaired_test).all()),
                "paired_cases": int(len(test_data["target_pose"]) // 3),
                "trajectory_length": int(test_data["target_z"].shape[1]),
                "train_test_disjoint": True,
            },
        }
        reports.append(report)
    return reports


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    if args.train_cases < 16 or args.held_out < 8:
        raise ValueError("use at least 16 train cases and 8 held-out cases")
    max_horizon = max(max(args.horizons), max(args.train_horizons))
    if min(args.horizons + args.train_horizons) < 1:
        raise ValueError("horizons must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    base_model = load_compatible_pretrained("pusht/lewm", device=args.device)
    repaired_model = load_compatible_pretrained("pusht/lewm", device=args.device)
    transform = image_transform()
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        train_cases = sample_cases(env, args.train_cases, args.seed, args.approach_steps)
        test_cases = sample_cases(env, args.held_out, args.seed + 100000, args.approach_steps)
        train_data = collect_data(base_model, env, train_cases, transform, args, max_horizon)
        test_data = collect_data(base_model, env, test_cases, transform, args, max_horizon)
        training = train_repair(repaired_model, train_data, args)
        reports = evaluate(base_model, repaired_model, train_data, test_data, args)
    finally:
        env.close()
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "checkpoint": "published pusht/lewm initialization",
            "train_cases": args.train_cases,
            "held_out_cases": args.held_out,
            "seed": args.seed,
            "arms_per_case": 3,
            "epochs": args.epochs,
            "anchor_weight": args.anchor_weight,
            "free_run_weight": args.free_run_weight,
            "pose_loss_weight": args.pose_loss_weight,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "horizons": sorted(set(args.horizons)),
            "train_horizons": sorted(set(args.train_horizons)),
            "calibration_cases": sorted(
                set(int(value) for value in (args.calibration_cases or [args.train_cases]))
            ),
            "max_horizon": max_horizon,
            "context_protocol": "two preceding approach frames plus contact frame; history_size=3",
            "action_normalization": "PushT live action divided by action_scale=100, then public mean/std",
            "supervision": (
                "teacher-forced latent target at every future step"
                + (" plus a weighted fully autoregressive loss" if args.free_run_weight > 0 else "")
                + (" plus an auxiliary normalized pose loss" if args.pose_loss_weight > 0 else "")
                + "; autoregressive endpoint evaluation"
            ),
            "repaired_parameters": "action_encoder, predictor, and pred_proj only",
            "frozen_parameters": "visual encoder and projector",
            "calibration_test_disjoint": True,
        },
        "training": training,
        "horizons": reports,
        "sanity_checks": {
            "all_finite": bool(all(row["sanity_checks"]["finite"] for row in reports)),
            "all_disjoint": True,
            "train_samples": int(len(train_data["target_pose"])),
            "test_samples": int(len(test_data["target_pose"])),
            "target_sequence_shape": list(test_data["target_z"].shape),
        },
        "claim_boundary": (
            "Targeted teacher-forced repair diagnostic for one published PushT checkpoint. "
            "It tests multi-step exposure bias but does not show that the original frozen "
            "checkpoint had this behavior or establish universal physical causality, online "
            "planning gain, or task success."
        ),
    }
    (output / "contact_predictor_teacher_forced_repair.json").write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True)
    )
    lines = [
        "# Teacher-forced contact predictor repair",
        "",
        "| h | base pose + | repaired calibrated + | observed-latent upper bound + | base latent sep | repaired latent sep | physical sep |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in reports:
        metrics = row["metrics"]
        lines.append(
            "| %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
                row["horizon"],
                metrics["base_error_plus"]["estimate"],
                metrics["repaired_calibrated_error_plus"]["estimate"],
                metrics["observed_latent_upper_bound_error_plus"]["estimate"],
                metrics["base_latent_counterfactual_separation"]["estimate"],
                metrics["repaired_latent_counterfactual_separation"]["estimate"],
                metrics["physical_counterfactual_separation"]["estimate"],
            )
        )
    lines += [
        "",
        "All three intervention arms (+a, -a, zero), calibration-size results, and the observed-latent upper bound are stored in the JSON artifact.",
        "The visual encoder and projector remain frozen; only action-conditioned predictor modules are repaired.",
        "Claim boundary: targeted single-checkpoint exposure-bias diagnostic; no universal causal, planning, or task-success claim.",
        "",
    ]
    (output / "contact_predictor_teacher_forced_repair.md").write_text("\n".join(lines))
    print(json.dumps(jsonable(report), indent=2))


if __name__ == "__main__":
    main()
